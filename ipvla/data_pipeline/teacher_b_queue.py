#!/usr/bin/env python3
"""Cut validated teacher A spans and enqueue isolated, resumable SOL causal checks."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import requests

from annotate_video_chunk_teacher_orx import cleanup_chat_session, collect, submit
from run_video_teacher_batch import BASE, read, write


def split_for(participant):
    # Preserve the earlier pilot's participant hold-outs.
    return "heldout" if participant.startswith(("2b40c765","5cb4f7ce","2641b427")) else "train"


def prepare_window(result, directory, ffmpeg):
    if (directory/"preflight.json").is_file():
        return
    if directory.exists():
        archived = directory.parent.parent / "interrupted_materialization"
        archived.mkdir(exist_ok=True)
        shutil.move(str(directory), str(archived / f"{directory.name}-{time.time_ns()}"))
    source = {}
    segments = []
    reviews = []
    for index, span in enumerate(result["segments"]):
        start = span["start_frame"]
        end = span["end_frame_exclusive"]
        segment_id = f"{result['job_id']}-s{index:03d}"
        segments.append(dict(segment_id=segment_id, start_frame=start, end_frame_exclusive=end,
                            executed_skill=span["executed_skill_en"], observed_outcome=span["observed_outcome_zh"],
                            annotation_kind=span["kind"], starts_in_progress=span["starts_in_progress"],
                            ends_in_progress=span["ends_in_progress"], role_a_message_id=result["message_id"]))
        if span["kind"] != "action":
            continue
        # Read up to the first tick with four real observations; never fabricate earlier frames.
        cutoff = max(start,3)
        if cutoff >= end:
            continue
        step = min(15, cutoff//3)
        frames = [cutoff-step*n for n in (3,2,1,0)]
        previous = result["segments"][index-1] if index else None
        alternative = previous["executed_skill_en"] if previous and previous["kind"]=="action" else "Inspect the visible work area before committing to another action."
        if alternative.casefold() == span["executed_skill_en"].casefold():
            alternative = "Inspect what is visible before committing to another action."
        reviews.append(dict(case_id=segment_id, segment_id=segment_id, cutoff_frame=cutoff,
                            frame_indices=frames, candidates={"a":span["executed_skill_en"],"b":alternative}))
    human = result.get("dataset") == "os_ost_part3"
    source.update(episode_id=result["episode_id"], participant_id=result["participant_id"],
                  split="unmapped_os_sequence" if human else split_for(result["participant_id"]), source_file=result["source_video"],
                  source_video_sha256=result["source_sha256"], source_member=result["source_member"],
                  goal=("Search for objects and bring them to the packing location." if human else
                        "Search the cluttered bin, retrieve objects and move them to the sorting box."),
                  segments=segments,reviews=reviews)
    spec = directory.with_suffix(".spec.json")
    write(spec,{"episodes":[source]})
    subprocess.run([sys.executable,str(Path(__file__).with_name("prepare_video_chunk_teacher.py")),
                    "--spec",str(spec),"--output",str(directory),"--ffmpeg",ffmpeg,"--single-pass"],check=True)


def record_failure(case, error, retry_limit):
    calls_dir = case / "role_b_openresearch"
    if calls_dir.is_dir():
        for receipt_file in calls_dir.glob("*.json"):
            try:
                rec = read(receipt_file)
                cleanup_chat_session(BASE, rec.get("chat_id"))
            except Exception:
                pass
    attempts = case / "attempts"
    attempts.mkdir(exist_ok=True)
    previous = len(list(attempts.iterdir()))
    archive = attempts / f"{previous+1:02d}"
    archive.mkdir()
    write(archive / "error.json", {"error": str(error), "failed_at_s": time.time()})
    for name in ("role_b_openresearch", "error.json"):
        if (case / name).exists():
            (case / name).replace(archive / name)
    if previous >= retry_limit:
        write(case / "error.json", {"error": str(error), "attempts": previous+1})


def poll_case(case, receipt_path, timeout_s):
    receipt = read(receipt_path)
    if collect(BASE, case, quiet=True):
        return "causal_checked"
    age = time.time() - receipt.get("requested_at_s", receipt_path.stat().st_mtime)
    if age > timeout_s:
        # Only this annotation worker's own stale session is interrupted.
        if "turn" in receipt:
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(BASE + f"/api/chat/sessions/{receipt['chat_id']}/interrupt", timeout=10)
                if response.status_code >= 500 or response.status_code == 429:
                    response.raise_for_status()
        raise TimeoutError(f"No complete teacher reply after {age:.0f} seconds")
    return "inflight"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--project",required=True)
    parser.add_argument("--ffmpeg",default="ffmpeg")
    parser.add_argument("--concurrency",type=int,default=2,choices=range(1,13))
    parser.add_argument("--materializers",type=int,default=1,choices=range(1,5))
    parser.add_argument("--case-timeout",type=int,default=360)
    parser.add_argument("--retry-limit",type=int,default=2)
    parser.add_argument("--retry-existing-errors",action="store_true")
    parser.add_argument("--montage",action="store_true",help="Send 4 causal RGB frames as one timestamped attachment")
    parser.add_argument("--wall-limit",type=int,default=2400)
    parser.add_argument("--follow",action="store_true",help="Wait for additional first-pass labels")
    args=parser.parse_args()
    output=args.output.resolve()
    windows=output/"materialized"
    sessions=output/"role_b_cases"
    windows.mkdir(parents=True,exist_ok=True)
    sessions.mkdir(exist_ok=True)
    begin=time.monotonic()
    if args.retry_existing_errors:
        for error in sessions.glob("*/error.json"):
            if not (error.parent / "attempts").exists():
                record_failure(error.parent, read(error)["error"], args.retry_limit)
    with ThreadPoolExecutor(max_workers=args.materializers) as materializer:
        pending = {}
        indexed = set()
        while time.monotonic()-begin < args.wall_limit:
            for window, future in list(pending.items()):
                if not future.done():
                    continue
                try:
                    future.result()
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    write(window.with_suffix(".error.json"), {"error": str(error)})
                del pending[window]
            phase_a = sorted((output/"role_a_results").glob("*.json"))
            for path in phase_a:
                window=windows/path.stem
                if window in indexed or window in pending:
                    continue
                ready = (window/"preflight.json").exists()
                if not ready and not window.with_suffix(".error.json").exists() and len(pending)<args.materializers:
                    pending[window] = materializer.submit(prepare_window, read(path), window, args.ffmpeg)
                if ready:
                    manifest=window/"role_b_packets.json"
                    source_binding=window/"offline_bindings.json"
                    bindings={row["case_id"]:row for row in read(source_binding)}
                    for packet in read(manifest):
                        case=sessions/packet["case_id"]
                        case.mkdir(exist_ok=True)
                        if not (case/"packet.json").exists():
                            write(case/"packet.json",packet)
                            write(case/"binding.json",bindings[packet["case_id"]])
                    indexed.add(window)
            counts=Counter()
            ep_checked=Counter()
            available=[]
            for case in sorted(sessions.iterdir()):
                if (case/"role_b_results.jsonl").exists():
                    counts["causal_checked"]+=1
                    ep_checked[case.name.split("-",1)[0]]+=1
                    continue
                if (case/"error.json").exists():
                    counts["needs_review"]+=1
                    continue
                requests_dir=case/"role_b_openresearch"
                receipts=list(requests_dir.glob("*.json")) if requests_dir.exists() else []
                if receipts:
                    try:
                        status = poll_case(case, receipts[0], args.case_timeout)
                        counts[status]+=1
                        if status == "causal_checked":
                            ep_checked[case.name.split("-",1)[0]]+=1
                    except requests.HTTPError as error:
                        if error.response is not None and error.response.status_code == 404:
                            print("Stale404Reset", case.name, flush=True)
                            shutil.rmtree(requests_dir, ignore_errors=True)
                            available.append(case)
                        else:
                            print(type(error).__name__, case.name, flush=True)
                            counts["inflight"] += 1
                    except requests.RequestException as error:
                        print(type(error).__name__,case.name,flush=True)
                        counts["inflight"]+=1
                    except (ValueError,TypeError,KeyError,IndexError,OSError) as error:
                        record_failure(case,error,args.retry_limit)
                        if (case/"error.json").exists():
                            counts["needs_review"]+=1
                        else:
                            available.append(case)
                else:
                    available.append(case)
            available.sort(key=lambda c: (ep_checked[c.name.split("-",1)[0]], c.name))
            for case in available:
                requests_dir=case/"role_b_openresearch"
                if counts["inflight"]>=args.concurrency:
                    counts["queued"]+=1
                    continue
                requests_dir.mkdir(exist_ok=True)
                try:
                    submit(BASE,args.project,requests_dir,read(case/"packet.json"),montage=args.montage)
                    counts["inflight"]+=1
                except requests.RequestException as error:
                    print(type(error).__name__,case.name,flush=True)
                    if list(requests_dir.glob("*.json")):
                        counts["inflight"]+=1
                    else:
                        counts["queued"]+=1
                    continue
                except (ValueError,TypeError,KeyError,IndexError,OSError) as error:
                    record_failure(case,error,args.retry_limit)
                    counts["needs_review" if (case/"error.json").exists() else "queued"]+=1
            write(output/"role_b_progress.json",dict(by_status=counts,local_gpu_hours=0,
                  materializing_windows=[window.name for window in pending],
                  indexed_windows=len(indexed)))
            apath=output/"role_a_progress.json"
            aphase=read(apath)["by_status"] if apath.exists() else {}
            if not args.follow and not pending and counts and not counts["queued"] and not counts["inflight"] and not aphase.get("queued") and not aphase.get("submitted"):
                break
            time.sleep(6)


if __name__=="__main__":
    main()
