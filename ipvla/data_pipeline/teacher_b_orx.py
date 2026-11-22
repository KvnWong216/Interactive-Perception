#!/usr/bin/env python3
"""Run each causal case in a fresh local OpenResearch GPT-6.2 SOL chat."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from pathlib import Path
import base64, hashlib, io, json, re, time
import requests
from PIL import Image, ImageDraw


def save_receipt(path, receipt):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def cleanup_chat_session(base, chat_id, attachment_paths=None):
    """Delete a finished/failed OpenResearch chat session to immediately kill its
    codex app-server + MCP child bundle, remove its git worktree, purge DB rows,
    and unlink temporary montage attachments."""
    if chat_id:
        try:
            with requests.Session() as session:
                session.trust_env = False
                resp = session.delete(base + f"/api/chat/sessions/{chat_id}", timeout=20)
                if resp.status_code not in (200, 204, 404):
                    print(f"cleanup warning {chat_id}: HTTP {resp.status_code}", flush=True)
        except Exception as exc:
            print(f"cleanup warning {chat_id}: {type(exc).__name__}: {exc}", flush=True)
    if attachment_paths:
        attachments_root = (Path.home() / ".local/share/openresearch/chat-attachments").resolve()
        for raw_path in attachment_paths:
            try:
                p = Path(raw_path).resolve()
                if p.is_relative_to(attachments_root) and p.is_file():
                    p.unlink()
            except OSError:
                pass


def submit(base, project, calls, packet, *, montage=False):
    case = packet['case_id']
    with requests.Session() as session:
        session.trust_env=False
        path = calls / f'{case}.json'
        if path.exists():
            previous = json.loads(path.read_text())
            chat_id = previous['chat_id']
            route_model = previous['route_model']
        else:
            response = session.post(base+'/api/chat/sessions', json={
                'projectId':project, 'harness':'codex','model':'gpt-6.2-sol','reasoningLevel':'high','permissionMode':'ask'
            }, timeout=15)
            response.raise_for_status()
            chat = response.json()['session']
            chat_id, route_model = chat['id'], chat['model']
        if route_model != 'gpt-6.2-sol':
            raise ValueError(f'Unexpected teacher route: {route_model}')
        images=[]
        image_hashes=[]
        tiles = []
        for n, frame in enumerate(packet['images']):
            source = Path(frame['path'])
            assert hashlib.sha256(source.read_bytes()).hexdigest()==frame['sha256']
            assert frame['pts_s'] <= packet['cutoff_pts_s']
            byte_stream=io.BytesIO()
            with Image.open(source) as image:
                if montage:
                    tile = image.convert('RGB')
                    tile.thumbnail((672, 504), Image.Resampling.LANCZOS)
                    tiles.append(tile)
                    continue
                image.convert('RGB').save(byte_stream,format='JPEG',quality=95,subsampling=0)
            image_bytes=byte_stream.getvalue()
            image_name=f'{case}__{n+1:02}__pts_{frame["pts_s"]:.3f}s.jpg'
            images.append({'mediaType':'image/jpeg','dataBase64':base64.b64encode(image_bytes).decode(),'name':image_name})
            image_hashes.append({'name':image_name,'source_sha256':frame['sha256'],'attached_sha256':hashlib.sha256(image_bytes).hexdigest(),'pts_s':frame['pts_s']})
        layout = 'four chronologically ordered RGB frames (oldest to newest; the 4th frame is the decision cutoff)'
        if montage:
            if len(tiles) != 4:
                raise ValueError("A causal montage requires four original frames")
            sheet = Image.new("RGB", (1344, 1048), "white")
            draw = ImageDraw.Draw(sheet)
            for i, (tile, frame) in enumerate(zip(tiles, packet["images"], strict=True)):
                x, y = (i % 2) * 672, (i // 2) * 524
                draw.text((x+4, y+3), f"{i+1}    PTS {frame['pts_s']:.3f}s", fill="black")
                sheet.paste(tile, (x, y+20))
            buffer = io.BytesIO()
            sheet.save(buffer,format="JPEG",quality=88)
            image_bytes = buffer.getvalue()
            sheet.close()
            for tile in tiles:
                tile.close()
            tiles.clear()
            buffer.close()
            name = f"{case}__4_causal_frames.jpg"
            images = [{"mediaType":"image/jpeg","dataBase64":base64.b64encode(image_bytes).decode(),"name":name}]
            image_hashes = [{"name":name,"attached_sha256":hashlib.sha256(image_bytes).hexdigest()}]
            layout = 'one 4-tile causal montage ordered 1 to 4 (top-left, top-right, bottom-left, bottom-right) from oldest to newest; inspect all 4 tiles, where tile 4 is the decision cutoff'
        payload = {'goal':packet['goal'],'cutoff_pts_s':packet['cutoff_pts_s'],
                   'candidates':{name:packet['candidates'][name] for name in ('b','a')}}
        prompt = f'''You are the independent causal vision Teacher B for IP-VLA. Evaluate ONLY this single case using the attached {layout}. Relying strictly on these causal images and the given goal, judge independently whether each candidate action is currently feasible and serves the goal at the cutoff frame. Each candidate is evaluated independently; both may be applicable. When the goal does not specify a particular object or slot, do not treat a single slot as mandatory. Use "unknown" if visual evidence is insufficient, and "inapplicable" only when there is clear visual evidence against feasibility. Distinguish between gripper contact, tentative grasp, and an object stably lifted clear of the bin. Do not inspect the workspace, other cases, full source videos, or external pages; do not run commands or write files. If attachment pixels are not visible, return images_seen=0 and mark both candidates unknown.

Case payload: {json.dumps(payload,ensure_ascii=False)}
Return ONLY a JSON object in English without markdown code fences: {{"images_seen":4,"results":{{"a":{{"applicability":"applicable|inapplicable|unknown","evidence":"concise English explanation citing visible evidence in the frames"}},"b":{{"applicability":"applicable|inapplicable|unknown","evidence":"concise English explanation citing visible evidence in the frames"}}}}}}.'''
        receipt = {'case_id':case,'chat_id':chat_id,'route_model':route_model,
                   'roles':'only isolated causal attachments, no executed segment outcome or source path',
                   'submitted_images':image_hashes,'input':payload,'prompt':prompt,
                   'requested_at_s':time.time(),'frame_count':len(packet['images']),
                   'attachment_layout':'four_tile_jpeg' if montage else 'separate_jpegs'}
        if montage:
            receipt['source_frames']=[{key:f[key] for key in ("sha256","pts_s","frame_index")} for f in packet['images']]
        save_receipt(path, receipt)
        response=session.post(base+f'/api/chat/sessions/{chat_id}/message',json={
            'text':prompt,'images':images,'clientTurnId':f'ip-vla-sol-chunks-v1-{case}'
        },timeout=25)
        response.raise_for_status()
        receipt['turn']=response.json()['turn']
        save_receipt(path, receipt)
        print(case,chat_id,len(images),flush=True)

def collect(base, output, *, quiet=False, cleanup=True):
    calls = output / "role_b_openresearch"
    records = []
    pending = []
    cleanup_targets = []
    allowed = {"applicable", "inapplicable", "unknown"}
    attachments = Path.home() / ".local/share/openresearch/chat-attachments"
    with requests.Session() as session:
        session.trust_env = False
        for file in sorted(calls.glob("*.json")):
            receipt = json.loads(file.read_text())
            case = receipt["case_id"]
            chat_id = receipt["chat_id"]
            response = session.get(base + f"/api/chat/sessions/{chat_id}/messages", timeout=15)
            response.raise_for_status()
            messages = response.json()["messages"]
            assistants = [message for message in messages if message["role"] == "assistant"]
            if assistants:
                errors = [part["state"] for part in assistants[-1]["parts"]
                          if part.get("type") == "tool" and part.get("tool") == "error"]
                if errors:
                    raise ValueError(f"Teacher turn failed: {errors[-1].get('error', 'unknown error')}")
            responses = [message for message in messages if message["role"] == "assistant" and message.get("completedAt")]
            if not responses:
                pending.append(case)
                continue
            latest = responses[-1]
            replies = [part["text"] for part in latest["parts"] if part["type"] == "text"]
            text = replies[-1] if replies else ""
            answer = json.loads(re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip()))
            tools = [part for message in responses for part in message["parts"]
                     if part["type"] == "tool" and part.get("tool") != "retry"]
            hashes = []
            reviewed_files = []
            for tool in tools:
                if tool["tool"] != "imageView" or tool["state"]["status"] != "completed":
                    raise ValueError(f"Unexpected evidence access: {case}")
                filename = Path(tool["state"]["input"]["path"]).resolve()
                if not filename.is_relative_to(attachments.resolve()):
                    raise ValueError(f"Image read outside OpenResearch attachments: {case}")
                hashes.append(hashlib.sha256(filename.read_bytes()).hexdigest())
                reviewed_files.append(filename)
            expected = {image["attached_sha256"] for image in receipt["submitted_images"]}
            if set(hashes) != expected or answer["images_seen"] != receipt.get("frame_count", len(expected)):
                raise ValueError(f"Incomplete or extra reviewed attachments: {case}")
            results = answer["results"]
            if results.keys() != receipt["input"]["candidates"].keys():
                raise ValueError(f"Candidate id mismatch: {case}")
            for value in results.values():
                if value["applicability"] not in allowed or not value["evidence"].strip():
                    raise ValueError(f"Invalid causal answer: {case}")
            records.append(dict(case_id=case, chat_id=chat_id, response_message_id=latest["id"],
                                annotator_model=receipt["route_model"], checked_attachment_sha256=hashes,
                                raw_text=text, results=results, annotation_stage="automatic_initial",
                                causal_only=True, user_approved=False,
                                attachment_layout=receipt.get("attachment_layout","separate_jpegs")))
            cleanup_targets.append((chat_id, reviewed_files))
    if pending:
        if not quiet:
            print(json.dumps({"completed": len(records), "pending": pending}, ensure_ascii=False))
        return False
    target = output / "role_b_results.jsonl"
    pending_target = target.with_suffix(".jsonl.tmp")
    with pending_target.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    try:
        target.hardlink_to(pending_target)
    finally:
        pending_target.unlink()
    if cleanup:
        for chat_id, reviewed_files in cleanup_targets:
            cleanup_chat_session(base, chat_id, reviewed_files)
    print(json.dumps({"completed": len(records), "output": str(target)}, ensure_ascii=False))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("submit", "collect"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project")
    parser.add_argument("--base", default="http://127.0.0.1:4791")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.mode == "submit":
        if not args.project:
            parser.error("--project is required in submit mode")
        calls = output / "role_b_openresearch"
        calls.mkdir(exist_ok=False)
        packets = json.loads((output / "role_b_packets.json").read_text())
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(submit, args.base, args.project, calls, packet) for packet in packets]
            for future in futures:
                future.result()
    elif not collect(args.base, output):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
