#!/usr/bin/env python3
"""Submit and resume a versioned, bounded-concurrency SOL video segmentation queue."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time

import av
from PIL import Image, ImageDraw
import requests


MODEL = "gpt-6.2-sol"
BASE = "http://127.0.0.1:4791"
ATTACHMENTS = Path.home() / ".local/share/openresearch/chat-attachments"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def recording(source, participant, member, source_hash, dataset, output, seconds):
    path = Path(source).resolve()
    key = source_hash[:16]
    directory = output / "media" / key
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "index.json"
    if index_path.exists():
        cached = read(index_path)
        if cached and cached[0]["source_sha256"] == source_hash:
            return cached
    if digest(path) != source_hash:
        raise ValueError(f"Changed source video: {path}")
    windows = []
    sampled = []
    first_time = next_sample = None
    count = 0

    def save_window(start, end, frames, anchor):
        frame_indices = [index for index, _, _ in frames]
        if not frames or frame_indices[0] != start:
            raise ValueError("First source frame missing from window")
        ident = f"{key}-{len(windows):03d}"
        sheets = []
        for j in range(0, len(frames), 15):
            group = frames[j:j+15]
            sheet = Image.new("RGB", (1200, 205 * ((len(group)+4)//5)), "white")
            draw = ImageDraw.Draw(sheet)
            for n, (index, pts, image) in enumerate(group):
                x,y = 240*(n%5),205*(n//5)
                draw.text((x+4,y+2), f"frame {index}  PTS {float(pts):.2f}s", fill="black")
                sheet.paste(image, (x,y+18))
            image_path = directory / f"{ident}-{j//15:02d}.jpg"
            sheet.save(image_path, quality=90, optimize=True)
            sheets.append(dict(path=str(image_path), sha256=digest(image_path)))
        return dict(job_id=ident, episode_id=key, participant_id=participant, source_video=str(path),
                    source_sha256=source_hash, source_member=member, dataset=dataset, start_frame=start,
                    end_frame_exclusive=end, visible_frames=frame_indices, start_pts_s=float(anchor),
                    sheets=sheets, model=MODEL, status="queued")

    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if frame.pts is None:
                raise ValueError(f"Frame without PTS: {path}:{index}")
            pts = frame.pts * frame.time_base
            if first_time is None:
                first_time = pts
                window_start = index
                window_time = pts
                next_sample = pts
                last_time = None
            if last_time is not None and pts <= last_time:
                raise ValueError(f"Nonmonotonic source PTS: {path}")
            if pts - window_time >= seconds:
                windows.append(save_window(window_start, index, sampled, window_time))
                sampled = []
                window_start = index
                window_time = pts
                next_sample = pts
            if pts >= next_sample:
                image = frame.to_image()
                image.thumbnail((240,180))
                sampled.append((index, pts, image))
                next_sample += 1
            last_time = pts
            count = index + 1
    if not count:
        raise ValueError(f"Empty recording: {path}")
    windows.append(save_window(window_start,count,sampled,window_time))
    write(index_path, windows)
    return windows


def submit(session, project, job, state_file):
    if job.get("chat_id"):
        chat_id = job["chat_id"]
    else:
        response = session.post(BASE+"/api/chat/sessions", json={"projectId":project,"harness":"codex",
            "model":MODEL,"reasoningLevel":"high","permissionMode":"ask"}, timeout=20)
        response.raise_for_status()
        chat = response.json()["session"]
        if chat["model"] != MODEL:
            raise ValueError(f"Unexpected teacher route: {chat['model']}")
        chat_id = chat["id"]
        job["chat_id"] = chat_id
        write(state_file,job)
    images = []
    for sheet in job["sheets"]:
        path = Path(sheet["path"])
        if digest(path) != sheet["sha256"]:
            raise ValueError("Changed annotation sheet")
        images.append({"mediaType":"image/jpeg","name":path.name,"dataBase64":base64.b64encode(path.read_bytes()).decode()})
    payload = {key:job[key] for key in ("start_frame","end_frame_exclusive","visible_frames")}
    setting = ("These are egocentric human first-person object-search frames. Label viewpoint search, clearing occlusion, inspection, grasping, and transport."
               if job.get("dataset") == "os_ost_part3" else
               "These are RoboTurk manipulation frames for searching and transferring objects from a cluttered bin to a sorting box. Label clearing occlusion, grasping, carrying/transporting, and aligning with a slot.")
    prompt = f"""You are Teacher A for IP-VLA video segmentation. Inspect all chronologically ordered contact sheets attached to this message and segment the window into contiguous half-open intervals by the semantic purpose of the visible action. The thumbnails are real frames sampled at roughly 1 second intervals from a continuous video region. {setting} Repeated grasp attempts on the same target may stay in one skill interval. Motion amplitude or fixed duration alone do not define boundaries. Distinguish successful, unsuccessful, and unclear outcomes. Do not invent specific target objects or slot names that are not visually evident. Mark intervals without clear action as unclear or idle rather than guessing a skill. If the first or last segment is already in progress at the window boundary, set starts_in_progress or ends_in_progress to true.
Use only the imageView tool to inspect every attached sheet. Do not call shell, Python, or image conversion tools, do not read the workspace or other cases, and do not write files. Segment boundaries must come from the frame indices printed on the sheets, the final end_frame_exclusive must equal the provided end_frame_exclusive, and intervals must form a contiguous partition.
Window metadata: {json.dumps(payload,ensure_ascii=False)}
Return ONLY a JSON object in English: {{"sheets_seen":{len(images)},"segments":[{{"start_frame":0,"end_frame_exclusive":1,"executed_skill_en":"short English imperative verb phrase or unclear","observed_outcome_en":"concise English description of the visible outcome","kind":"action|idle|unclear","starts_in_progress":false,"ends_in_progress":false}}]}}. Replace the example numbers with real boundaries."""
    turn = session.post(BASE+f"/api/chat/sessions/{chat_id}/message",json={"text":prompt,"images":images,
        "clientTurnId":"ip-vla-teacher-a-full-v1-"+job["job_id"]},timeout=45)
    turn.raise_for_status()
    job.update(status="submitted",turn=turn.json()["turn"],submitted_at_s=time.time())
    write(state_file,job)
    print("submitted",job["job_id"],chat_id,flush=True)


def collect(session, job, state_file, results, timeout_s):
    response = session.get(BASE+f"/api/chat/sessions/{job['chat_id']}/messages",timeout=15)
    response.raise_for_status()
    messages = response.json()["messages"]
    completed = [m for m in messages if m["role"]=="assistant" and m.get("completedAt")]
    if not completed:
        assistants = [m for m in messages if m["role"] == "assistant"]
        errors = [p["state"] for p in assistants[-1]["parts"]
                  if p.get("type") == "tool" and p.get("tool") == "error"] if assistants else []
        age = time.time() - job.get("submitted_at_s", state_file.stat().st_mtime)
        if errors or age > timeout_s:
            if not errors:
                session.post(BASE + f"/api/chat/sessions/{job['chat_id']}/interrupt", timeout=10).raise_for_status()
            error = errors[-1].get("error", "Teacher turn failed") if errors else f"Teacher timeout after {age:.0f}s"
            job.update(status="needs_review",error=error)
            write(state_file,job)
        return
    message = completed[-1]
    replies = [p["text"] for p in message["parts"] if p["type"]=="text"]
    text = replies[-1] if replies else ""
    tools = [p for m in completed for p in m["parts"] if p["type"]=="tool"]
    actual = set()
    try:
        for tool in tools:
            if tool["tool"] == "error":
                raise ValueError("Teacher turn failed: " + str(tool["state"].get("error", "no message"))[:800])
            if tool["tool"] != "imageView" or tool["state"]["status"] != "completed":
                raise ValueError(f"Unexpected tool {tool['tool']}")
            path = Path(tool["state"]["input"]["path"]).resolve()
            if not path.is_relative_to(ATTACHMENTS.resolve()):
                raise ValueError("Teacher viewed a file outside attachments")
            actual.add(digest(path))
        if actual != {sheet["sha256"] for sheet in job["sheets"]}:
            raise ValueError("Missing or extra viewed attachments")
        body = json.loads(re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$","",text.strip()))
        if body["sheets_seen"] != len(job["sheets"]):
            raise ValueError("Teacher did not confirm every sheet")
        previous = job["start_frame"]
        boundaries = {*job["visible_frames"],job["end_frame_exclusive"]}
        for span in body["segments"]:
            start,end = span["start_frame"],span["end_frame_exclusive"]
            if start != previous or end not in boundaries or end <= start:
                raise ValueError("Bad segmentation coverage")
            if span["kind"] not in {"action","idle","unclear"} or not span["executed_skill_en"].strip():
                raise ValueError("Bad segment label")
            outcome_en = span.get("observed_outcome_en") or span.get("observed_outcome_zh") or ""
            span["observed_outcome_en"] = outcome_en
            span["observed_outcome_zh"] = outcome_en
            if any(not isinstance(span[flag],bool) for flag in ("starts_in_progress","ends_in_progress")):
                raise ValueError("Missing continuation status")
            previous = end
        if previous != job["end_frame_exclusive"]:
            raise ValueError("Segmentation does not end at assigned boundary")
    except (KeyError, TypeError, ValueError, OSError) as error:
        job.update(status="needs_review",error=str(error),message_id=message["id"],raw_text=text)
    else:
        job.update(status="segmented",message_id=message["id"])
        write(results/f"{job['job_id']}.json",dict(job, segments=body["segments"], raw_text=text,
            annotation_stage="automatic_initial",causal_language_supervision_approved=False))
    write(state_file,job)
    print(job["status"],job["job_id"],job.get("error",""),flush=True)


def retry(job, file, limit):
    failures = job.setdefault("failed_attempts", [])
    if len(failures) >= limit:
        return
    failures.append({key:job.pop(key) for key in ("chat_id","turn","submitted_at_s","message_id","error","raw_text") if key in job})
    job["status"] = "queued"
    write(file,job)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True,type=Path)
    parser.add_argument("--additional-manifest",action="append",type=Path,default=[])
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--project",required=True)
    parser.add_argument("--concurrency",type=int,default=3,choices=range(1,9))
    parser.add_argument("--case-timeout",type=int,default=450)
    parser.add_argument("--retry-limit",type=int,default=2)
    parser.add_argument("--wall-limit",type=int,default=1800)
    parser.add_argument("--follow",action="store_true",help="Wait for additional verified video ingests")
    args=parser.parse_args()
    begin=time.monotonic()
    state=args.output/"state"
    results=args.output/"role_a_results"
    state.mkdir(parents=True,exist_ok=True)
    results.mkdir(exist_ok=True)
    registered = set()
    failed_sources = args.output / "source_preflight_errors"
    failed_sources.mkdir(exist_ok=True)
    pending = None
    with requests.Session() as session, ThreadPoolExecutor(max_workers=1) as decoder:
        session.trust_env=False
        while time.monotonic()-begin < args.wall_limit:
            if pending and pending[1].done():
                source_hash, future = pending
                try:
                    for job in future.result():
                        file = state / f"{job['job_id']}.json"
                        if not file.exists():
                            write(file, job)
                except (OSError, ValueError) as error:
                    write(failed_sources/f"{source_hash}.json", {"error":str(error)})
                    print("failed to decode",source_hash,error,flush=True)
                pending = None
            if pending is None:
                manifests = [args.manifest, *(p for p in args.additional_manifest if p.exists())]
                for row in (item for manifest in manifests for item in read(manifest)):
                    if not row.get("source_video") or row["sha256"] in registered:
                        continue
                    sha = row["sha256"]
                    registered.add(sha)
                    if (failed_sources/f"{sha}.json").exists():
                        continue
                    cached = args.output/"media"/sha[:16]/"index.json"
                    if cached.exists():
                        windows=read(cached)
                        for job in windows:
                            file=state/f"{job['job_id']}.json"
                            if not file.exists():
                                write(file,job)
                        continue
                    pending = (sha, decoder.submit(recording,row["source_video"],row["participant_id"],
                               row["source_member"],sha,row["dataset"],args.output,45))
                    break
            jobs=[(p,read(p)) for p in sorted(state.glob("*.json"))]
            inflight=sum(job["status"]=="submitted" for _,job in jobs)
            for file,job in jobs:
                try:
                    if job["status"]=="submitted":
                        collect(session,job,file,results,args.case_timeout)
                        if job["status"]!="submitted":
                            inflight-=1
                            if job["status"]=="needs_review":
                                retry(job,file,args.retry_limit)
                    elif job["status"]=="queued" and inflight<args.concurrency:
                        submit(session,args.project,job,file)
                        inflight+=1
                except requests.HTTPError as error:
                    status=error.response.status_code
                    try:
                        detail=error.response.json().get("error", "")
                    except (ValueError, AttributeError):
                        detail=""
                    endpoint=error.request.path_url.split("?",1)[0]
                    job["http_diagnostic"]=dict(method=error.request.method, endpoint=endpoint,
                                                status=status, server_error=str(detail)[:800])
                    if status == 429 or status >= 500:
                        print(f"Transient HTTP {status} for {job['job_id']}; retrying later",flush=True)
                    else:
                        job.update(status="needs_review",error=f"HTTP {status}")
                        write(file,job)
                except requests.RequestException as error:
                    print(type(error).__name__,job["job_id"],flush=True)
            counts=Counter(read(p)["status"] for p,_ in jobs)
            write(args.output/"role_a_progress.json",dict(by_status=counts,local_gpu_hours=0,teacher_model=MODEL))
            if not args.follow and pending is None and not counts["queued"] and not counts["submitted"]:
                break
            time.sleep(5)


if __name__=="__main__":
    main()
