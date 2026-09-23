"""Isolated simulator process for an ordinary LIBERO control smoke check."""

import argparse
import json
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from libero_protocol import applied_prefix
from prepare_libero import (
    LIBERO,
    ROOT,
    configure,
    inside,
    model_xml,
    public_frame,
    restore,
)


def render_profile_xml(xml, profile):
    """Change only offscreen sample count; retain native rendering by default."""
    if profile == "native":
        return xml
    if profile != "no_msaa":
        raise ValueError(f"unsupported render profile: {profile}")
    root = ET.fromstring(xml)
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    quality = visual.find("quality")
    if quality is None:
        quality = ET.SubElement(visual, "quality")
    quality.set("offsamples", "0")
    return ET.tostring(root, encoding="unicode")


def create_environment(
    source, episode_name, manual_seed, physical_gpu=7, render_profile="native"
):
    """Restore a recorded physical state with calibrated cameras on the chosen GPU."""
    configure()
    from gpu_devices import configure_render_gpu

    configure_render_gpu(physical_gpu)
    from libero.libero.envs import TASK_MAPPING

    source = inside(source)
    with h5py.File(source, "r") as archive:
        data, episode = archive["data"], archive[f"data/{episode_name}"]
        config = json.loads(data.attrs["env_args"])
        xml, initial = model_xml(episode.attrs["model_file"]), episode["states"][0]
        task = json.loads(data.attrs["problem_info"])["language_instruction"]
    xml = render_profile_xml(xml, render_profile)
    kwargs = config["env_kwargs"].copy()
    kwargs.update(
        bddl_file_name=str(
            LIBERO
            / "bddl_files"
            / source.parent.name
            / (source.stem.removesuffix("_demo") + ".bddl")
        ),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_depths=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        camera_segmentations=None,
        ignore_done=True,
        render_gpu_device_id=physical_gpu,
    )
    env = TASK_MAPPING[config["problem_name"]](**kwargs)
    try:
        env.seed(manual_seed)
        env.reset()
        env.reset_from_xml_string(xml)
        env.sim.reset()
        raw = restore(env, initial)
        if render_profile == "no_msaa":
            context = env.sim._render_context_offscreen
            if env.sim.model.vis.quality.offsamples != 0 or context.con.offSamples != 0:
                raise RuntimeError("offscreen multisampling was not disabled")
        return env, raw, task
    except BaseException:
        env.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manual-seed", type=int, default=17)
    parser.add_argument("--physical-gpu", type=int, default=7)
    args = parser.parse_args()
    output = inside(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env, raw, task = create_environment(
        args.file, args.episode, args.manual_seed, args.physical_gpu
    )
    try:
        step = 0
        applied = []
        while True:
            frame = public_frame(env, raw)
            destination = output / f"observation_{step:04d}.npz"
            np.savez_compressed(destination, **frame)
            # Evaluation success is a separate evaluator field. The client
            # never places it in the VLA observation or task prompt.
            print(
                "IP_OBSERVATION:"
                + json.dumps(
                    {
                        "task": task,
                        "step": step,
                        "path": str(destination.relative_to(ROOT)),
                        "evaluation_success": bool(env._check_success()),
                        "applied_actions": applied,
                    }
                ),
                flush=True,
            )
            line = sys.stdin.readline()
            if not line:
                break
            request = json.loads(line)
            if request.get("close"):
                break
            actions = applied_prefix(request["actions"], *env.action_spec)
            applied = []
            for action in actions:
                env.step(action)
                applied.append({"step": step, "value": action.tolist()})
                step += 1
            env._update_observables(force=True)
            raw = env._get_observations()
    finally:
        env.close()


if __name__ == "__main__":
    main()
