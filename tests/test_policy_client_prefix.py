import numpy as np
import pytest

from interactive_perception.policy_client import (
    ObservationPacket,
    OpenPiWebsocketPolicy,
)


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.payload = None

    def infer(self, payload):
        self.payload = payload
        return self.response


def packet() -> ObservationPacket:
    return ObservationPacket(
        image=np.zeros((224, 224, 3), dtype=np.uint8),
        wrist_image=np.zeros((224, 224, 3), dtype=np.uint8),
        state=np.zeros(8, dtype=np.float32),
        prompt="Place the butter in the basket",
    )


def policy_with(fake: FakeClient) -> OpenPiWebsocketPolicy:
    policy = object.__new__(OpenPiWebsocketPolicy)
    policy.host = "fake"
    policy.port = 0
    policy.api_key = None
    policy._client = fake
    return policy


def test_prefix_request_is_explicit_and_has_frozen_shape() -> None:
    fake = FakeClient({"prefix_features": np.ones(8192, dtype=np.float32)})
    value = policy_with(fake).encode_prefix(packet())
    assert value.shape == (8192,)
    assert fake.payload["__request_type"] == "prefix"
    assert fake.payload["__feature_schema"] == "global_v1"


def test_cognitive_spatial_v5_request_has_frozen_shape() -> None:
    fake = FakeClient({"prefix_features": np.ones(21504, dtype=np.float32)})
    value = policy_with(fake).encode_prefix(
        packet(), feature_schema="cognitive_spatial_v5"
    )
    assert value.shape == (21504,)
    assert fake.payload["__feature_schema"] == "cognitive_spatial_v5"


def test_stock_server_is_rejected_for_prefix_request() -> None:
    with pytest.raises(RuntimeError, match="extended server"):
        policy_with(FakeClient({"actions": np.zeros((10, 7))})).encode_prefix(packet())


def test_full_spatial_prefix_response_preserves_every_token_and_span() -> None:
    response = {
        "schema_version": "piu.spatial-prefix-response.v1",
        "camera_names": ["base_0_rgb", "left_wrist_0_rgb"],
        "tokens_per_camera": [4, 4],
        "image_tokens": np.ones((8, 16), dtype=np.float16),
        "image_valid_mask": np.ones(8, dtype=bool),
        "prompt_tokens": np.ones((5, 16), dtype=np.float16),
        "prompt_valid_mask": np.ones(5, dtype=bool),
    }
    fake = FakeClient(response)
    value = policy_with(fake).encode_spatial_prefix(packet())
    assert value["image_tokens"].shape == (8, 16)
    assert value["prompt_tokens"].shape == (5, 16)
    assert value["tokens_per_camera"] == (4, 4)
    assert fake.payload["__feature_schema"] == "spatial_prefix_v1"
