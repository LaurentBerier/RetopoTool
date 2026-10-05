"""Low-level glTF/GLB helpers shared by the decimator, the bake and the measurement tools.

  - `_acc`: integer-aware accessor reader (any componentType, honours byteStride).
  - `_resize_image`: Pillow downscale of an embedded texture, with a decompression-bomb guard.
  - DoS bounds for untrusted input (`MAX_IMAGE_PIXELS`, `MAX_TOTAL_VERTS`).
"""
from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np
from pygltflib import GLTF2

logger = logging.getLogger(__name__)

_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_TN = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

# DoS bounds for untrusted GLB input: cap embedded-texture pixels (decompression-bomb guard) and total
# vertices (decimation is super-linear). Generous vs real rigged characters (~1M verts / 2–4k px
# textures); they only reject pathological/malicious inputs.
MAX_IMAGE_PIXELS = 64_000_000          # 64 MP — bail before decoding a decompression bomb
MAX_TOTAL_VERTS = 5_000_000            # ~5M verts across all primitives


def _acc(g: GLTF2, blob: bytes, idx: int) -> np.ndarray:
    """Integer-aware accessor reader (handles any componentType + byteStride)."""
    a = g.accessors[idx]
    bv = g.bufferViews[a.bufferView]
    dt = np.dtype(_CT[a.componentType])
    n = _TN[a.type]
    base = (bv.byteOffset or 0) + (a.byteOffset or 0)
    stride = bv.byteStride or dt.itemsize * n
    if stride == dt.itemsize * n:
        flat = np.frombuffer(blob, dtype=dt, count=a.count * n, offset=base)
        return flat.reshape(a.count, n) if n > 1 else flat.copy()
    out = np.empty((a.count, n), dtype=dt)
    for i in range(a.count):
        out[i] = np.frombuffer(blob, dtype=dt, count=n, offset=base + i * stride)
    return out if n > 1 else out.reshape(-1)


def _resize_image(raw: bytes, mime: str, max_dim: int) -> Optional[bytes]:
    """Resize an embedded texture to fit max_dim (keeps aspect). Returns None if already small enough."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS   # Pillow raises DecompressionBombError past this on decode
    img = Image.open(io.BytesIO(raw))           # lazy — header only, no full decode yet
    w, h = img.size
    if w * h > MAX_IMAGE_PIXELS:                 # reject by header dimensions BEFORE the decode allocates
        raise ValueError(f"texture too large to optimize safely: {w}x{h}")
    if max(w, h) <= max_dim:
        return None
    scale = max_dim / float(max(w, h))
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    out = io.BytesIO()
    if "jpeg" in mime or "jpg" in mime:
        img.convert("RGB").save(out, format="JPEG", quality=90)
    else:
        img.save(out, format="PNG", optimize=True)
    return out.getvalue()
