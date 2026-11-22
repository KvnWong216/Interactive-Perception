"""Zero-Shot Interactive Perception (ZS-IP, arXiv:2602.18374) — Enhanced Observation & Memory Module.

Implements the two non-trainable perception/reasoning pillars of Sripada, Guerin, & Ghalamzan (2026)
under strict zero-oracle ActiveArena rules (uses ONLY `camera_head` RGB + calibrated URDF extrinsics/intrinsics):
  1. **Enhanced Observation (EO)**:
     - Projects calibrated 2D fan-workspace spatial grid anchors (`G1..G6`) onto the image;
     - Detects tabletop foreground regions via non-oracle visual contrast + homography plane ray-casting
       (`z_pad = 0.742 m`, `z_obj = 0.778 m`), labeling grasp/place **Keypoints (`K1..Km`)** with dominant
       color descriptors and workspace coordinates, while masking proprioceptive robot gripper projections;
     - Generates contact-rich **Pushline affordances (`P1->D1 .. Pn->Dn`)** mapping pre-contact `P_i`
       to post-contact `D_i` trajectories for occlusion-clearing pushes.
  2. **Memory-Guided Context Lookup (`ZSIPMemory`)**:
     - Stores chronological primitive execution records, grasped object origin coordinates `p_origin`,
       and pinned high-resolution visual reveal frames captured after `INSPECT_HELD` / `PUSH`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


TABLE_Z_M = 0.742
OBJECT_RAY_Z_M = 0.778
GRASP_TCP_Z_M = 0.766
PLACE_TCP_Z_M = 0.782
FAN_INNER_R_M = 0.22
FAN_OUTER_R_M = 0.76


@dataclass(frozen=True)
class CameraPinhole:
    width: int = 512
    height: int = 384
    fovy_deg: float = 60.0
    sensor_aspect: float = 4.0 / 3.0

    @property
    def fy(self) -> float:
        return (self.height * 0.5) / math.tan(math.radians(self.fovy_deg) * 0.5)

    @property
    def fx(self) -> float:
        # On native 4:3 (512x384), fx == fy; on resized square crops (e.g. 224x224), preserves exact horizontal FOV.
        return (self.width * 0.5) / (math.tan(math.radians(self.fovy_deg) * 0.5) * self.sensor_aspect)

    @property
    def cx(self) -> float:
        return self.width * 0.5

    @property
    def cy(self) -> float:
        return self.height * 0.5

    def project_world_to_pixel(self, p_world: np.ndarray, T_world_cam: np.ndarray) -> tuple[float, float] | None:
        """Project 3D Sapien world point into `camera_head` FLU frame (`+X` forward, `+Y` left, `+Z` up)."""
        inv_cam = np.linalg.inv(T_world_cam)
        pw = np.asarray(p_world, dtype=np.float64).reshape(3)
        pc = inv_cam[:3, :3] @ pw + inv_cam[:3, 3]
        if pc[0] <= 0.05:
            return None
        u = self.cx - self.fx * (pc[1] / pc[0])
        v = self.cy - self.fy * (pc[2] / pc[0])
        return float(u), float(v)

    def raycast_pixel_to_z_plane(
        self, u: float, v: float, T_world_cam: np.ndarray, z_plane: float = TABLE_Z_M
    ) -> np.ndarray | None:
        """Intersect pixel `(u, v)` ray with horizontal tabletop plane `z = z_plane` in Sapien world."""
        d_cam = np.array(
            [1.0, -(float(u) - self.cx) / self.fx, -(float(v) - self.cy) / self.fy],
            dtype=np.float64,
        )
        R_wc = T_world_cam[:3, :3]
        p_wc = T_world_cam[:3, 3]
        d_world = R_wc @ d_cam
        if abs(d_world[2]) < 1e-5:
            return None
        lam = (float(z_plane) - p_wc[2]) / d_world[2]
        if lam <= 0.05 or lam > 2.5:
            return None
        return p_wc + lam * d_world


@dataclass
class KeypointAnchor:
    kid: str
    pixel_uv: tuple[int, int]
    world_xyz: np.ndarray
    grasp_xyz: np.ndarray
    place_xyz: np.ndarray
    color_tag: str
    area_px: int
    kind: str  # "near_zone" | "mid_object_zone" | "outer_clutter_zone"


@dataclass
class PushlineAffordance:
    pid: str
    target_kid: str
    pre_world_xyz: np.ndarray
    post_world_xyz: np.ndarray
    pre_uv: tuple[int, int]
    post_uv: tuple[int, int]


@dataclass
class EnhancedObservationResult:
    annotated_rgb: np.ndarray
    keypoints: dict[str, KeypointAnchor]
    pushlines: dict[str, PushlineAffordance]
    grid_anchors: dict[str, np.ndarray]
    text_catalog: str


def _classify_patch_color(rgb_patch: np.ndarray) -> str:
    hsv = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2HSV)
    sat = float(np.median(hsv[..., 1]))
    val = float(np.median(hsv[..., 2]))
    hue = float(np.median(hsv[..., 0]))
    if sat < 36:
        if val < 88:
            return "dark/black"
        if val < 208:
            return "gray"
        return "white"
    if hue < 10 or hue >= 168:
        return "red"
    if hue < 22:
        return "orange"
    if hue < 36:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 132:
        return "blue"
    return "purple"


def extract_proprio_reveal_crop(
    rgb: np.ndarray,
    tcp_world_xyz: np.ndarray,
    T_world_cam: np.ndarray,
    crop_size: int = 256,
    pinhole: CameraPinhole | None = None,
) -> np.ndarray:
    """Crop clean `camera_head` RGB around proprioceptively projected holding gripper TCP `(u, v)`.

    Uses strictly robot forward kinematics (`tcp_world_xyz` & `T_world_cam`) with zero object oracle,
    giving the frozen VLM a clean, un-occluded view of the inspected backside patch or date label.
    """
    h, w = rgb.shape[:2]
    cam = pinhole or CameraPinhole(width=w, height=h)
    uv = cam.project_world_to_pixel(tcp_world_xyz, T_world_cam)
    if uv is None:
        cu, cv_y = w // 2, h // 2
    else:
        cu = int(np.clip(round(uv[0]), 0, w - 1))
        cv_y = int(np.clip(round(uv[1]), 0, h - 1))
    half = min(crop_size, w, h) // 2
    x0 = int(np.clip(cu - half, 0, max(0, w - 2 * half)))
    y0 = int(np.clip(cv_y - half, 0, max(0, h - 2 * half)))
    return np.asarray(rgb[y0 : y0 + 2 * half, x0 : x0 + 2 * half], dtype=np.uint8).copy()


def build_enhanced_observation(
    rgb: np.ndarray,
    T_world_cam: np.ndarray,
    T_world_robot0: np.ndarray,
    pinhole: CameraPinhole | None = None,
    max_keypoints: int = 8,
    exclude_uvs: list[tuple[float, float]] | None = None,
    exclude_radius_px: int = 44,
) -> EnhancedObservationResult:
    """Construct ZS-IP Enhanced Observation (2D Grid + Foreground Keypoints + Pushlines)."""
    h, w = rgb.shape[:2]
    cam = pinhole or CameraPinhole(width=w, height=h)
    robot_xy = T_world_robot0[:2, 3]
    fwd_xy = T_world_robot0[:2, 0]   # +Y_world
    left_xy = T_world_robot0[:2, 1]  # -X_world

    # 1. Build 6 canonical 2D spatial fan-grid anchors `G1..G6`
    grid_specs = [
        ("G1", 0.39, +0.40),  # near-left (pad zone)
        ("G2", 0.39, 0.00),   # near-center (pad zone)
        ("G3", 0.39, -0.40),  # near-right (pad zone)
        ("G4", 0.48, +0.48),  # mid-left (object zone)
        ("G5", 0.46, 0.00),   # mid-center (object zone)
        ("G6", 0.48, -0.48),  # mid-right (object zone)
    ]
    grid_anchors: dict[str, np.ndarray] = {}
    grid_uvs: dict[str, tuple[int, int]] = {}
    for gid, r_m, theta_rad in grid_specs:
        xy = robot_xy + r_m * (math.cos(theta_rad) * fwd_xy + math.sin(theta_rad) * left_xy)
        p_w = np.array([xy[0], xy[1], PLACE_TCP_Z_M], dtype=np.float64)
        grid_anchors[gid] = p_w
        uv = cam.project_world_to_pixel(np.array([xy[0], xy[1], TABLE_Z_M]), T_world_cam)
        if uv is not None and 8 <= uv[0] < w - 8 and 8 <= uv[1] < h - 8:
            grid_uvs[gid] = (int(round(uv[0])), int(round(uv[1])))

    # 2. Non-oracle foreground segmentation on tabletop ray-valid pixels
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Tabletop & robot white links are bright pure white (`S < 28, V > 208`); colored pads, gray blocks, cans & clutter differ
    non_white = (hsv[..., 1] >= 28) | (hsv[..., 2] <= 206)

    yy, xx = np.mgrid[0:h:4, 0:w:4]
    valid_coarse = np.zeros_like(xx, dtype=np.uint8)
    for iy in range(xx.shape[0]):
        for ix in range(xx.shape[1]):
            pw = cam.raycast_pixel_to_z_plane(float(xx[iy, ix]), float(yy[iy, ix]), T_world_cam, TABLE_Z_M)
            if pw is not None:
                rel = pw[:2] - robot_xy
                r_dist = float(np.hypot(rel[0], rel[1]))
                fwd_proj = float(np.dot(rel, fwd_xy))
                if FAN_INNER_R_M <= r_dist <= FAN_OUTER_R_M and fwd_proj >= -0.02:
                    valid_coarse[iy, ix] = 1
    table_mask = cv2.resize(valid_coarse, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    fg_mask = (non_white & table_mask).astype(np.uint8) * 255
    if exclude_uvs:
        rad_scaled = max(16, int(round(exclude_radius_px * (w / 512.0))))
        for eu, ev in exclude_uvs:
            if 0 <= eu < w and 0 <= ev < h:
                cv2.circle(fg_mask, (int(round(eu)), int(round(ev))), rad_scaled, 0, thickness=-1)

    k_sz = max(3, int(round(5 * (w / 512.0))) | 1)
    kernel = np.ones((k_sz, k_sz), dtype=np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    min_area = max(45, int(round(130 * ((w * h) / (512.0 * 384.0)))))
    num_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    cand_blobs: list[tuple[int, float, float, str, int]] = []
    for lbl_i in range(1, num_lbl):
        area = int(stats[lbl_i, cv2.CC_STAT_AREA])
        if area < min_area or area > int(0.25 * w * h):
            continue
        cu, cv_y = float(centroids[lbl_i, 0]), float(centroids[lbl_i, 1])
        x0, y0, bw, bh = [
            int(stats[lbl_i, k])
            for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)
        ]
        patch = rgb[max(0, y0) : min(h, y0 + bh), max(0, x0) : min(w, x0 + bw)]
        blob_mask = labels[max(0, y0) : min(h, y0 + bh), max(0, x0) : min(w, x0 + bw)] == lbl_i
        if np.any(blob_mask):
            med_rgb = patch[blob_mask].reshape(-1, 1, 3)
            ctag = _classify_patch_color(med_rgb)
        else:
            ctag = _classify_patch_color(patch)
        # Skip robot's own peripheral black gripper fingers when resting near image margins
        if ctag == "dark/black" and (cu < 0.14 * w or cu > 0.86 * w) and cv_y < 0.52 * h:
            continue
        cand_blobs.append((area, cu, cv_y, ctag, lbl_i))

    cand_blobs.sort(key=lambda x: (-x[0]))
    cand_blobs = cand_blobs[:max_keypoints]
    cand_blobs.sort(key=lambda x: x[1])

    keypoints: dict[str, KeypointAnchor] = {}
    pushlines: dict[str, PushlineAffordance] = {}
    p_idx = 1

    for idx, (area, cu, cv_y, ctag, _) in enumerate(cand_blobs, 1):
        pw_table = cam.raycast_pixel_to_z_plane(cu, cv_y, T_world_cam, z_plane=TABLE_Z_M)
        pw_obj = cam.raycast_pixel_to_z_plane(cu, cv_y, T_world_cam, z_plane=OBJECT_RAY_Z_M)
        if pw_table is None or pw_obj is None:
            continue
        kid = f"K{idx}"
        grasp_xyz = np.array([pw_obj[0], pw_obj[1], GRASP_TCP_Z_M], dtype=np.float64)
        place_xyz = np.array([pw_table[0], pw_table[1], PLACE_TCP_Z_M], dtype=np.float64)

        rel = pw_table[:2] - robot_xy
        r_dist = float(np.hypot(rel[0], rel[1]))
        if r_dist < 0.405 and ctag != "gray":
            kind = "near_pad_or_object"
        elif r_dist <= 0.54:
            kind = "mid_workspace_object"
        else:
            kind = "outer_clutter_candidate"

        keypoints[kid] = KeypointAnchor(
            kid=kid,
            pixel_uv=(int(round(cu)), int(round(cv_y))),
            world_xyz=pw_table,
            grasp_xyz=grasp_xyz,
            place_xyz=place_xyz,
            color_tag=ctag,
            area_px=area,
            kind=kind,
        )

        # Generate ZS-IP Pushline `P_i -> D_i` for movable mid/outer candidates
        if kind in {"mid_workspace_object", "outer_clutter_candidate"} and p_idx <= 4:
            lat_dir = left_xy if float(np.dot(rel, left_xy)) >= 0.0 else -left_xy
            pre_xy = grasp_xyz[:2] - 0.075 * lat_dir
            post_xy = grasp_xyz[:2] + 0.085 * lat_dir
            pre_w = np.array([pre_xy[0], pre_xy[1], GRASP_TCP_Z_M + 0.010], dtype=np.float64)
            post_w = np.array([post_xy[0], post_xy[1], GRASP_TCP_Z_M + 0.010], dtype=np.float64)
            uv_pre = cam.project_world_to_pixel(pre_w, T_world_cam)
            uv_post = cam.project_world_to_pixel(post_w, T_world_cam)
            if uv_pre is not None and uv_post is not None:
                pid = f"P{p_idx}"
                pushlines[pid] = PushlineAffordance(
                    pid=pid,
                    target_kid=kid,
                    pre_world_xyz=pre_w,
                    post_world_xyz=post_w,
                    pre_uv=(int(round(uv_pre[0])), int(round(uv_pre[1]))),
                    post_uv=(int(round(uv_post[0])), int(round(uv_post[1]))),
                )
                p_idx += 1

    # 3. Render Enhanced Observation visual overlay onto image copy
    pil_im = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(pil_im)

    for gid, (gu, gv) in grid_uvs.items():
        draw.rectangle([gu - 12, gv - 8, gu + 12, gv + 8], outline=(80, 140, 220), width=1)
        draw.text((gu - 9, gv - 6), gid, fill=(30, 90, 190))

    for pid, pl in pushlines.items():
        u1, v1 = pl.pre_uv
        u2, v2 = pl.post_uv
        draw.line([u1, v1, u2, v2], fill=(0, 210, 210), width=2)
        draw.ellipse([u2 - 3, v2 - 3, u2 + 3, v2 + 3], fill=(0, 210, 210))
        draw.text((u1 - 6, v1 - 13), f"{pid}->D{pid[1:]}", fill=(0, 130, 150))

    for kid, kp in keypoints.items():
        ku, kv = kp.pixel_uv
        col = (220, 40, 40) if kp.kind != "near_pad_or_object" else (30, 160, 60)
        draw.ellipse([ku - 8, kv - 8, ku + 8, kv + 8], outline=col, width=2)
        draw.rectangle([ku + 7, kv - 9, ku + 27, kv + 5], fill=(0, 0, 0))
        draw.text((ku + 9, kv - 7), kid, fill=(255, 255, 100))

    lines = ["Detected Enhanced Observation Anchors:"]
    for kid, kp in keypoints.items():
        rel = kp.world_xyz[:2] - robot_xy
        r_m = float(np.hypot(rel[0], rel[1]))
        az_deg = math.degrees(math.atan2(float(np.dot(rel, left_xy)), float(np.dot(rel, fwd_xy))))
        lines.append(
            f"  - {kid}: color={kp.color_tag}, zone={kp.kind}, radius={r_m:.2f}m, azimuth={az_deg:+.0f}deg "
            f"(world_xy=({kp.world_xyz[0]:+.2f}, {kp.world_xyz[1]:+.2f}))"
        )
    for pid, pl in pushlines.items():
        lines.append(f"  - {pid}->D{pid[1:]}: pushline across {pl.target_kid}")
    lines.append(
        "  - Workspace Grid Sectors: G1(near-left), G2(near-center), G3(near-right), "
        "G4(mid-left), G5(mid-center), G6(mid-right)"
    )

    return EnhancedObservationResult(
        annotated_rgb=np.asarray(pil_im, dtype=np.uint8),
        keypoints=keypoints,
        pushlines=pushlines,
        grid_anchors=grid_anchors,
        text_catalog="\n".join(lines),
    )


@dataclass
class ZSIPMemory:
    """Memory-guided context lookup for ZS-IP (`arXiv:2602.18374`)."""

    max_reveal_frames: int = 2
    holding: bool = False
    holding_arm: str = "left"
    origin_xyz: list[float] | None = None
    inspected_current_object: bool = False
    scanned_directions: list[str] = field(default_factory=list)
    action_history: list[dict[str, Any]] = field(default_factory=list)
    reveal_frames: list[np.ndarray] = field(default_factory=list)

    def reset(self) -> None:
        self.holding = False
        self.holding_arm = "left"
        self.origin_xyz = None
        self.inspected_current_object = False
        self.scanned_directions.clear()
        self.action_history.clear()
        self.reveal_frames.clear()

    def record_action(self, turn: int, primitive: str, target: str, note: str = "") -> None:
        self.action_history.append(
            {"turn": turn, "primitive": primitive, "target": target, "note": note}
        )

    def pin_reveal_frame(self, rgb: np.ndarray) -> None:
        self.reveal_frames.append(np.asarray(rgb, dtype=np.uint8).copy())
        if len(self.reveal_frames) > self.max_reveal_frames:
            self.reveal_frames = self.reveal_frames[-self.max_reveal_frames :]

    def format_context_lookup(self) -> str:
        hold_txt = (
            f"HOLDING object in {self.holding_arm} gripper (picked from {self.origin_xyz}, "
            f"backside_inspected={self.inspected_current_object})"
            if self.holding
            else "EMPTY grippers (not holding any object)"
        )
        scanned_txt = ", ".join(self.scanned_directions) if self.scanned_directions else "none"
        hist_lines = [
            f"  Turn {h['turn']}: {h['primitive']}({h['target']}) -> {h['note']}"
            for h in self.action_history[-8:]
        ] or ["  (No primitives executed yet)"]
        return (
            f"Gripper State: {hold_txt}\n"
            f"Previously Scanned Directions: {scanned_txt}\n"
            f"Pinned Inspection/Reveal Frames in Memory: {len(self.reveal_frames)}\n"
            f"Past Interaction History:\n" + "\n".join(hist_lines)
        )
