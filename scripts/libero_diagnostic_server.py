"""Controlled camera shift and real-history blackout probes, isolated from VLA."""

import argparse
import json
import sys

import numpy as np
from libero_probe_server import create_environment
from libero_protocol import applied_prefix
from prepare_libero import ROOT, check_depth_rays, inside, public_frame, write_json


def refresh(env):
    env.sim.forward()
    env._update_observables(force=True)
    return env._get_observations()


def shift_camera(env, yaw_degrees, pitch_degrees):
    from robosuite.utils.transform_utils import mat2quat, quat2mat

    model = env.sim.model
    camera = model.camera_name2id("agentview")
    if model.cam_bodyid[camera] != 0:
        raise ValueError("camera perturbation requires a world-attached camera")
    rotation = quat2mat(np.roll(model.cam_quat[camera], -1))

    def rotate(axis, degrees):
        axis = np.asarray(axis) / np.linalg.norm(axis)
        x, y, z = axis
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        angle = np.deg2rad(degrees)
        return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross

    delta = rotate([0, 0, 1], yaw_degrees) @ rotate(rotation[:, 0], pitch_degrees)
    center = np.array([0, 0, 0.95])
    model.cam_pos[camera] = center + delta @ (model.cam_pos[camera] - center)
    model.cam_quat[camera] = np.roll(mat2quat(delta @ rotation), 1)


def object_position(env, name):
    return np.array(env.sim.data.body_xpos[env.obj_body_id[name]])


def move_target(env, target, offset):
    """Translate one existing free object; return its exact state coordinates."""
    joint = env.objects_dict[target].joints[0]
    pose = np.array(env.sim.data.get_joint_qpos(joint))
    if pose.shape != (7,):
        raise ValueError("target must be a free-joint object")
    pose[:3] += np.asarray(offset, dtype=float)
    env.sim.data.set_joint_qpos(joint, pose)
    env.sim.data.set_joint_qvel(joint, np.zeros(6))
    joint_id = env.sim.model.joint_name2id(joint)
    qpos = int(env.sim.model.jnt_qposadr[joint_id])
    qvel = int(env.sim.model.jnt_dofadr[joint_id])
    return [*range(1+qpos, 1+qpos+7),
            *range(1+env.sim.model.nq+qvel, 1+env.sim.model.nq+qvel+6)]


def target_contacts(env, target):
    """Evaluator-only geometric checks; no contact truth enters the policy."""
    body = env.obj_body_id[target]
    descendants = {body}
    for _ in range(env.sim.model.nbody):
        expanded = descendants | {i for i, parent in enumerate(env.sim.model.body_parentid)
                                   if parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    distances = []
    for contact in env.sim.data.contact[:env.sim.data.ncon]:
        a = int(env.sim.model.geom_bodyid[contact.geom1]) in descendants
        b = int(env.sim.model.geom_bodyid[contact.geom2]) in descendants
        if a != b:
            distances.append(float(contact.dist))
    return {"min_contact_distance_m": min(distances, default=0.0),
            "contact_count": len(distances)}


def prepare_history(env, case, output):
    """Change a scene before observation; every past frame/control is real."""
    ignored_state_indices = []
    setup = case.get("history_setup", "two_bowls_color")
    if setup == "two_bowls_color":
        names = ["akita_black_bowl_1", "akita_black_bowl_2"]
        target = names[case["target_index"]]
        distractor = names[1 - case["target_index"]]
        for name in env.objects_dict[distractor].visual_geoms:
            geom = env.sim.model.geom_name2id(name)
            env.sim.model.geom_matid[geom] = -1
            env.sim.model.geom_rgba[geom] = [0.9, 0.9, 0.9, 1]
        positions = [object_position(env, name) for name in names]
    elif setup in {"single_bowl_position", "single_object_position"}:
        target = case.get("target_object", "akita_black_bowl_1")
        center = object_position(env, target)
        positions = [center + np.array([0, sign * 0.06, 0]) for sign in (-1, 1)]
        ignored_state_indices = move_target(env, target, positions[case["target_index"]]-center)
    else:
        raise ValueError("unknown history scene setup")
    raw = refresh(env)
    hover = np.mean(positions, axis=0)
    hover[2] = max(p[2] for p in positions) + 0.15
    history, controls = [], []
    for step in range(41):
        if step % 5 == 0:
            frame = public_frame(env, raw)
            path = output / f"prefix_{step:04d}.npz"
            np.savez_compressed(path, **frame)
            history.append({"step": step, "path": str(path.relative_to(ROOT))})
        if step == 40:
            break
        action = np.zeros(7, dtype=np.float32)
        scale = np.asarray(env.robots[0].controller.output_max[:3])
        action[:3] = np.clip((hover - raw["robot0_eef_pos"]) / scale, -1, 1)
        action[-1] = -1
        action = applied_prefix([action], *env.action_spec)[0]
        env.step(action)
        controls.append({"step": step, "value": action.tolist()})
        raw = refresh(env)
    distance = float(np.linalg.norm(raw["robot0_eef_pos"] - hover))
    if distance > 0.015:
        raise RuntimeError(
            f"scripted neutral hover did not reach its target: {distance}"
        )
    axis = positions[1][:2] - positions[0][:2]
    if np.linalg.norm(axis) < 0.08:
        raise ValueError("bowl targets are too close for a directional decision")
    axis /= np.linalg.norm(axis)
    # Retain the two previous observations, not a second copy of current.
    public_history = {"observations": history[-3:-1], "applied_actions": controls[-10:]}
    private = {
        "target": target,
        "target_index": case["target_index"],
        "axis_xy": axis.tolist(),
        "takeover_eef": raw["robot0_eef_pos"].tolist(),
        "hover_error_m": distance,
        "target_positions": [p.tolist() for p in positions],
        "history_setup": setup,
        "ignored_state_indices": ignored_state_indices,
        "target_final_position": object_position(env, target).tolist(),
        "target_contacts": target_contacts(env, target),
    }
    return raw, public_history, private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--physical-gpu", type=int, default=7)
    args = parser.parse_args()
    case = json.loads(inside(args.case).read_text())
    output = inside(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env, raw, task = create_environment(
        ROOT / case["source"], case["episode"], case["manual_seed"], args.physical_gpu
    )
    try:
        history, private, first_choice = None, {}, None
        if case["kind"] in {"G", "S"}:
            shift_camera(env, case["yaw_degrees"], case["pitch_degrees"])
            if case["kind"] == "S":
                light_scale = float(case["light_scale"])
                if not 0.4 <= light_scale <= 1.0:
                    raise ValueError("unsupported lighting severity")
                for name in ("light_diffuse", "light_ambient", "light_specular"):
                    values = getattr(env.sim.model, name)
                    values[:] *= light_scale
            raw = refresh(env)
            step = 0
        elif case["kind"] == "H":
            raw, history, private = prepare_history(env, case, output)
            if private["history_setup"] != "single_object_position":
                task = (
                    "Put the bowl on the plate."
                    if private["history_setup"] == "single_bowl_position"
                    else "Put the black bowl on the plate."
                )
            step = 40
        elif case["kind"] == "P":
            target = case["target_object"]
            before = object_position(env, target)
            excluded = move_target(env, target, case["target_offset_m"])
            raw = refresh(env)
            private = {"target": target, "ignored_state_indices": excluded,
                       "target_before": before.tolist(),
                       "target_after": object_position(env, target).tolist(),
                       "target_contacts": target_contacts(env, target)}
            step = 0
        else:
            raise ValueError("unknown diagnostic kind")
        takeover = step
        calibration = check_depth_rays(env, public_frame(env, raw))
        write_json(
            output / "scene_audit.json",
            {
                "case": case,
                "calibration": calibration,
                "private_evaluation": private,
                "physical_state": env.sim.get_state().flatten().tolist(),
            },
        )
        applied = []
        while True:
            frame = public_frame(env, raw)
            if (
                case["kind"] == "H"
                and case["condition"] == "hidden"
                and step == takeover
            ):
                # Sensor blackout for one observation only; depth is unknown.
                for view in ("agent", "wrist"):
                    frame[f"{view}_rgb"][:] = 128
                    frame[f"{view}_depth"][:] = 0
            destination = output / f"observation_{step:04d}.npz"
            np.savez_compressed(destination, **frame)
            success = (
                bool(env._eval_predicate(["on", private["target"], "plate_1"]))
                if case["kind"] == "H" and private["history_setup"] != "single_object_position"
                else bool(env._check_success())
            )
            print(
                "IP_OBSERVATION:"
                + json.dumps(
                    {
                        "task": task,
                        "step": step,
                        "path": str(destination.relative_to(ROOT)),
                        "applied_actions": applied,
                        "initial_history": history if step == takeover else None,
                        "evaluation_success": success,
                        "evaluation_first_choice": first_choice,
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
            if len(actions) != 5:
                raise ValueError(
                    "diagnostic cases require five-step execution prefixes"
                )
            applied = []
            for action in actions:
                env.step(action)
                applied.append({"step": step, "value": action.tolist()})
                step += 1
            raw = refresh(env)
            if case["kind"] == "H" and first_choice is None:
                displacement = (
                    np.asarray(raw["robot0_eef_pos"])[:2]
                    - np.asarray(private["takeover_eef"])[:2]
                )
                projection = float(np.dot(displacement, private["axis_xy"]))
                choice = None if abs(projection) <= 0.002 else int(projection > 0)
                first_choice = {
                    "chosen_index": choice,
                    "correct": choice == case["target_index"],
                    "axis_displacement_m": projection,
                    "measured_at_step": step,
                }
    finally:
        env.close()


if __name__ == "__main__":
    main()
