"""Content-addressed public RGB storage used by concrete integrations.

The formal contracts intentionally carry image identities rather than file
paths.  This module provides the missing concrete resolver.  Its digest is over
decoded, canonical ``uint8`` RGB pixels plus their shape, so it is stable across
PNG encoder versions.  PNG is only the lossless storage container.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

from .contracts import PublicFrame


def _require_image_dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:  # pragma: no cover - depends on optional extra.
        raise RuntimeError(
            "RGB integration requires numpy and Pillow; install the integration extra"
        ) from error
    return np, Image


def canonical_rgb_array(value: Any) -> Any:
    """Return a detached contiguous HWC ``uint8`` RGB array.

    Floating-point inputs and implicit channel conversion are rejected.  The
    simulator/collector, rather than this audit boundary, owns such transforms.
    """

    np, _ = _require_image_dependencies()
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("RGB image must have shape [height, width, 3]")
    if array.dtype != np.uint8:
        raise TypeError("RGB image must have dtype uint8")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("RGB image dimensions must be positive")
    return np.ascontiguousarray(array).copy()


def canonical_rgb_sha256(value: Any) -> str:
    """Hash canonical RGB pixels and shape, independent of PNG metadata."""

    array = canonical_rgb_array(value)
    height, width, channels = (int(item) for item in array.shape)
    header = f"rgb8-v1\n{height}\n{width}\n{channels}\n".encode("ascii")
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def encode_rgb_png(value: Any) -> bytes:
    """Encode one validated RGB array as a lossless PNG byte string."""

    array = canonical_rgb_array(value)
    _, Image = _require_image_dependencies()
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=6,
    )
    return buffer.getvalue()


def decode_rgb_png(value: bytes) -> Any:
    """Decode a PNG while rejecting animation and non-RGB output ambiguity."""

    np, Image = _require_image_dependencies()
    if not isinstance(value, bytes) or not value:
        raise TypeError("PNG payload must be non-empty bytes")
    with Image.open(io.BytesIO(value)) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("animated images are not valid public RGB frames")
        image.load()
        rgb = image.convert("RGB")
        return canonical_rgb_array(np.asarray(rgb, dtype=np.uint8))


class RGBFrameStore:
    """Resolve :class:`PublicFrame` values against a content-addressed tree."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def _path(self, digest: str) -> Path:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("frame digest must be lowercase SHA-256")
        return self.root / "sha256" / digest[:2] / f"{digest}.png"

    def put(
        self,
        value: Any,
        *,
        frame_id: str,
        camera: str,
        frame_index: int,
    ) -> PublicFrame:
        """Store one public image and return the exact metadata contract."""

        array = canonical_rgb_array(value)
        digest = canonical_rgb_sha256(array)
        path = self._path(digest)
        encoded = encode_rgb_png(array)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = decode_rgb_png(path.read_bytes())
            if canonical_rgb_sha256(existing) != digest:
                raise ValueError(f"corrupt existing frame at {path}")
        else:
            path.write_bytes(encoded)
        height, width = array.shape[:2]
        return PublicFrame(
            frame_id=frame_id,
            camera=camera,
            frame_index=frame_index,
            image_sha256=digest,
            width=int(width),
            height=int(height),
        )

    def resolve(self, frame: PublicFrame) -> Any:
        """Load and fully revalidate the RGB pixels named by ``frame``."""

        if not isinstance(frame, PublicFrame):
            raise TypeError("frame must be a PublicFrame")
        path = self._path(frame.image_sha256)
        if not path.is_file():
            raise FileNotFoundError(f"public frame is absent from frame store: {path}")
        array = decode_rgb_png(path.read_bytes())
        height, width = array.shape[:2]
        if (int(width), int(height)) != (frame.width, frame.height):
            raise ValueError("decoded frame dimensions do not match PublicFrame")
        if canonical_rgb_sha256(array) != frame.image_sha256:
            raise ValueError("decoded RGB digest does not match PublicFrame")
        return array

    def path_for(self, frame: PublicFrame) -> Path:
        """Return a verified local path for reporting or visualization only."""

        self.resolve(frame)
        return self._path(frame.image_sha256)
