"""Low-level glTF/GLB helpers shared by the decimator, the bake and the measurement tools.

  - `_acc`: integer-aware accessor reader (any componentType, honours byteStride).
  - `load_triangles` / `concat_triangles`: every triangle primitive, placed by its node transform,
    for the measurement tools.
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
_TN = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}

# DoS bounds for untrusted GLB input: cap embedded-texture pixels (decompression-bomb guard) and total
# vertices (decimation is super-linear). Generous vs real rigged characters (~1M verts / 2–4k px
# textures); they only reject pathological/malicious inputs.
MAX_IMAGE_PIXELS = 64_000_000          # 64 MP — bail before decoding a decompression bomb
MAX_TOTAL_VERTS = 5_000_000            # ~5M verts across all primitives


def _acc(g: GLTF2, blob: bytes, idx: int) -> np.ndarray:
    """Integer-aware accessor reader (handles any componentType + byteStride)."""
    a = g.accessors[idx]
    dt = np.dtype(_CT[a.componentType])
    n = _TN[a.type]
    if a.bufferView is None:          # glTF: an accessor with no bufferView is all zeros
        z = np.zeros((a.count, n), dtype=dt)
        return z if n > 1 else z.reshape(-1)
    bv = g.bufferViews[a.bufferView]
    base = (bv.byteOffset or 0) + (a.byteOffset or 0)
    stride = bv.byteStride or dt.itemsize * n
    if stride == dt.itemsize * n:
        flat = np.frombuffer(blob, dtype=dt, count=a.count * n, offset=base)
        return flat.reshape(a.count, n) if n > 1 else flat.copy()
    # Interleaved vertex data (several attributes sharing one strided bufferView): a strided VIEW
    # over the blob, copied once. Reading it element by element in Python takes minutes on a
    # million-vertex scan.
    if a.count and base + (a.count - 1) * stride + dt.itemsize * n > len(blob):
        raise ValueError(f"accessor {idx} runs past the end of the binary chunk")
    out = np.ndarray((a.count, n), dtype=dt, buffer=blob, offset=base,
                     strides=(stride, dt.itemsize)).copy()
    return out if n > 1 else out.reshape(-1)


def _node_matrix(node) -> np.ndarray:
    if node.matrix is not None and len(node.matrix) == 16:
        return np.asarray(node.matrix, dtype=np.float64).reshape(4, 4).T     # column-major
    M = np.eye(4)
    if node.scale is not None:
        M = np.diag(list(node.scale) + [1.0]) @ M
    if node.rotation is not None:
        x, y, z, w = node.rotation
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        T = np.eye(4)
        T[:3, :3] = R
        M = T @ M
    if node.translation is not None:
        T = np.eye(4)
        T[:3, 3] = node.translation
        M = T @ M
    return M


def mesh_instances(g: GLTF2):
    """[(mesh index, world matrix)] for every node that draws a mesh, walking the default scene
    (or every scene). A skinned node's own transform is ignored, as glTF specifies; a mesh no node
    references is reported once at identity, so nothing in the file goes unmeasured."""
    out = []
    seen = set()
    scenes = g.scenes or []
    if g.scene is not None and g.scene < len(scenes):
        scenes = [scenes[g.scene]]
    roots = [r for sc in scenes for r in (sc.nodes or [])]
    if not roots and g.nodes:
        child = {c for n in g.nodes for c in (n.children or [])}
        roots = [i for i in range(len(g.nodes)) if i not in child]
    stack = [(r, np.eye(4), 0) for r in roots]
    while stack:
        ni, parent, depth = stack.pop()
        if depth > 256 or ni >= len(g.nodes or []):
            continue
        node = g.nodes[ni]
        M = parent @ _node_matrix(node)
        if node.mesh is not None:
            out.append((node.mesh, np.eye(4) if node.skin is not None else M))
            seen.add(node.mesh)
        for c in (node.children or []):
            stack.append((c, M, depth + 1))
    for mi in range(len(g.meshes or [])):
        if mi not in seen:
            out.append((mi, np.eye(4)))
    return out


def load_triangles(path: str, world: bool = True):
    """Every triangle primitive in a GLB, ready to measure: a list of dicts with P, F (int64),
    UV / NORMAL / TANGENT (float, or None), JOINTS_0 / WEIGHTS_0 (or None), `material` and `mesh`.
    With `world=True` each mesh instance is placed by its node transform (normals and tangents
    rotated by the inverse-transpose), so a multi-object prop is measured as it is assembled."""
    g = GLTF2().load(path)
    blob = g.binary_blob()
    if blob is None:
        raise ValueError(f"{path} has no binary chunk; it is not a self-contained GLB")

    def fl(idx, force=False):
        if idx is None:
            return None
        arr = _acc(g, blob, idx)
        if np.issubdtype(arr.dtype, np.integer) and (force or g.accessors[idx].normalized):
            arr = arr.astype(np.float64) / np.iinfo(arr.dtype).max
        return arr.astype(np.float64)

    prims = []
    insts = mesh_instances(g) if world else [(i, np.eye(4)) for i in range(len(g.meshes or []))]
    for mi, M in insts:
        for prim in g.meshes[mi].primitives:
            at = prim.attributes
            if at.POSITION is None or (prim.mode if prim.mode is not None else 4) != 4:
                continue
            P = fl(at.POSITION)
            F = (_acc(g, blob, prim.indices).astype(np.int64).reshape(-1, 3)
                 if prim.indices is not None else np.arange(len(P), dtype=np.int64).reshape(-1, 3))
            N, T = fl(at.NORMAL), fl(at.TANGENT)
            if not np.allclose(M, np.eye(4)):
                P = P @ M[:3, :3].T + M[:3, 3]
                Nm = np.linalg.inv(M[:3, :3]).T
                if N is not None:
                    N = N @ Nm.T
                    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-20
                if T is not None:
                    T = np.concatenate([T[:, :3] @ M[:3, :3].T, T[:, 3:4]], axis=1)
                if np.linalg.det(M[:3, :3]) < 0:          # mirrored instance: keep faces outward
                    F = F[:, ::-1]
            prims.append({
                "mesh": mi, "material": prim.material, "P": P, "F": F,
                "UV": fl(at.TEXCOORD_0), "NORMAL": N, "TANGENT": T,
                "JOINTS_0": (_acc(g, blob, at.JOINTS_0).astype(np.int64)
                             if getattr(at, "JOINTS_0", None) is not None else None),
                "WEIGHTS_0": fl(getattr(at, "WEIGHTS_0", None)),
            })
    return g, blob, prims


def concat_triangles(prims, keys=("UV", "NORMAL", "TANGENT", "JOINTS_0", "WEIGHTS_0")):
    """Concatenate `load_triangles` output; an attribute is kept only if EVERY primitive has it."""
    if not prims:
        raise ValueError("no triangle geometry")
    off, Fs = 0, []
    for p in prims:
        Fs.append(p["F"] + off)
        off += len(p["P"])
    out = {"P": np.concatenate([p["P"] for p in prims]), "F": np.concatenate(Fs)}
    for k in keys:
        out[k] = (np.concatenate([p[k] for p in prims])
                  if all(p[k] is not None for p in prims) else None)
    return out


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
