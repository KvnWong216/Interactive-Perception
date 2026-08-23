from __future__ import annotations

import numpy as np

from interactive_perception.oracle_visual_prompt import (
    PromptBox,
    build_oracle_target_packet,
    render_visual_prompt,
    target_box,
)
from interactive_perception.policy_client import build_observation


def observation(*, visible: bool = True) -> dict[str, np.ndarray]:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 1] = 80
    segmentation = np.zeros((16, 16, 1), dtype=np.int32)
    if visible:
        segmentation[4:9, 6:11, 0] = 7
    return {
        "agentview_image": image.copy(),
        "robot0_eye_in_hand_image": image.copy(),
        "agentview_segmentation_instance": segmentation.copy(),
        "robot0_eye_in_hand_segmentation_instance": segmentation.copy(),
        "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.asarray([0.02, -0.02]),
        "hidden_target_pose": np.ones(7),
    }


def test_target_box_is_half_open_padded_and_clipped() -> None:
    mask = np.zeros((8, 10), dtype=bool)
    mask[0:2, 8:10] = True
    assert target_box(mask, padding=3) == PromptBox(5, 0, 10, 5)
    assert target_box(np.zeros_like(mask)) is None


def test_all_visual_prompt_styles_preserve_rgb_contract() -> None:
    source = np.full((20, 20, 3), 120, dtype=np.uint8)
    box = PromptBox(5, 6, 13, 15)
    for style in ("box", "point", "spotlight"):
        rendered = render_visual_prompt(source, box, style=style, line_width=2)
        assert rendered.shape == source.shape
        assert rendered.dtype == source.dtype
        assert np.any(np.all(rendered == np.asarray([255, 0, 255]), axis=-1))
        assert np.array_equal(source, np.full_like(source, 120))


def test_oracle_mask_is_rendered_but_never_serialized() -> None:
    result = build_oracle_target_packet(
        observation(),
        "Place the marked butter in the basket",
        target_instance_id=7,
        style="box",
    )
    payload = result.packet.to_openpi()
    assert set(payload) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }
    assert result.diagnostics.visible_pixels == {
        "agentview": 25,
        "robot0_eye_in_hand": 25,
    }
    assert result.diagnostics.boxes["agentview"] == PromptBox(2, 0, 15, 13)
    assert np.any(
        np.all(payload["observation/image"] == np.asarray([255, 0, 255]), axis=-1)
    )


def test_invisible_target_is_an_audited_noop() -> None:
    raw = observation(visible=False)
    result = build_oracle_target_packet(
        raw,
        "Place the butter in the basket",
        target_instance_id=7,
        style="spotlight",
    )
    stock = build_observation(raw, "Place the butter in the basket")
    assert result.diagnostics.visible_pixels == {
        "agentview": 0,
        "robot0_eye_in_hand": 0,
    }
    assert result.diagnostics.boxes == {
        "agentview": None,
        "robot0_eye_in_hand": None,
    }
    assert np.array_equal(result.packet.image, stock.image)
    assert np.array_equal(result.packet.wrist_image, stock.wrist_image)
