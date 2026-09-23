"""Explain completed history decisions using real recorded inputs and endpoints.

Presentation holds are explicitly labelled; these are not real-time replays.
"""

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/stage1_expanded_s3217"
OUTPUT = ROOT / "docs/assets/stage1_expanded"


def main():
    objects, reports = {}, {}
    for gpu in (0,4,5,6,7):
        phase = RUN / f"gpu{gpu}/H"
        spec = json.loads((phase / "audit/eligible/manifest.json").read_text())
        if not spec["cases"]:
            continue
        report = json.loads((phase / "evaluation/report.json").read_text())
        assert report["complete"] and report["modes"] == ["native","best"]
        for case in spec["cases"]:
            objects.setdefault(case["target_object"],{})[case["layout"]] = phase
        reports[phase] = { (r["policy"],r["case_id"]):r for r in report["episodes"] }
    rng = np.random.default_rng(3317)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",17)
    big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",22)
    summaries = []
    for obj, layouts in sorted(objects.items()):
        layout = str(rng.choice(sorted(layouts)))
        phase = layouts[layout]
        rows = [reports[phase][p,f"{layout}_target{t}_hidden"] for t in (0,1) for p in ("native","best")]
        directories = [phase / "evaluation" / r["policy"] / r["case_id"] / r["episode"] for r in rows]
        frames = []
        for stage in ("past","current","endpoint"):
            canvas = Image.new("RGB",(1024,736),"#132234")
            draw = ImageDraw.Draw(canvas)
            draw.text((16,10),f"{obj} | {layout} | hidden-current decision",font=big)
            title = {"past":"A. Real past evidence, step 35 (history input to best)",
                     "current":"B. Current input, step 40: identical across target worlds",
                     "endpoint":"C. After the first 5 actions: evaluator view, not decision input"}[stage]
            draw.text((16,48),title,font=font)
            if stage in {"past","current"}:
                for world,index in ((0,0),(1,2)):
                    name = "prefix_0035.npz" if stage == "past" else "observation_0040.npz"
                    with np.load(directories[index]/name,allow_pickle=False) as frame:
                        for j,view in enumerate(("agent","wrist")):
                            canvas.paste(Image.fromarray(frame[view+"_rgb"]),(world*512+j*256,132))
                    draw.text((world*512+10,98),f"Target world {world} (evaluator label)",font=font)
                    draw.text((world*512+10,406),"agent view             wrist view",font=font)
                text = ("Best uses past steps 30/35 and current 40. Native uses current 40 only."
                        if stage=="past" else "Same current RGB-D, proprioception, instruction and paired sampling seed.")
                draw.text((16,474),text,font=font)
            else:
                for i,(row,directory) in enumerate(zip(rows,directories)):
                    fc = row["first_choice"]
                    draw.text((i*256+8,94),f"world {i//2} | {row['policy']}",font=font)
                    with np.load(directory/"observation_0045.npz",allow_pickle=False) as frame:
                        for j,view in enumerate(("agent","wrist")):
                            canvas.paste(Image.fromarray(frame[view+"_rgb"]),(i*256,142+j*270))
                    status = "correct" if fc["correct"] else "not met"
                    draw.text((i*256+8,120),f"{status}; dy={1000*fc['axis_displacement_m']:+.2f} mm",font=font)
            draw.text((16,690),"Presentation holds, not real-time replay. Endpoint: direction >2 mm, not task success.",font=font)
            frames.append(np.asarray(canvas))
        filename = f"history_{obj}.mp4"
        with imageio.get_writer(OUTPUT/filename,fps=4,codec="libx264",quality=7,ffmpeg_params=["-threads","2"]) as writer:
            for frame,count in zip(frames,(8,8,16)):
                for _ in range(count):
                    writer.append_data(frame)
        Image.fromarray(frames[2]).save(OUTPUT/f"history_{obj}_endpoint.jpg")
        summaries.append({"object":obj,"layout":layout,"file":filename,"selection_manual_seed":3317,
                          "selection":"one admissible layout per object; independent of policy scores",
                          "source":str(phase.relative_to(ROOT)),
                          "decisions":[{"policy":r["policy"],"case_id":r["case_id"],"first_choice":r["first_choice"]} for r in rows]})
    (OUTPUT/"history_demos.json").write_text(json.dumps(summaries,indent=2)+"\n")
    print(json.dumps({"videos":len(summaries),"output":str(OUTPUT.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
