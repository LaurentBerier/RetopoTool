"""Synthetic ENVIRONMENT / PROP GLBs: hard edges, several materials per mesh, shared materials and
accessors, tiling UVs, flat floors. The character fixtures in `_fixtures.py` are one smooth
primitive with one atlas; most props are none of those things."""
import io

import numpy as np
from PIL import Image
from pygltflib import (GLTF2, Accessor, BufferView, Buffer, Mesh, Primitive, Attributes, Node,
                       Scene, Material, PbrMetallicRoughness, Image as GLTFImage, Texture,
                       NormalMaterialTexture, TextureInfo, FLOAT, UNSIGNED_INT)

from retopotool.gltf_io import _CT


def _grid(n, relief=0.02, seed=0):
    """(n x n) grid over [0,1]^2 with a smooth bump field. Returns (uv01 (n*n,2), h (n*n,), F)."""
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="xy")
    rng = np.random.default_rng(seed)
    fx, fy, ph = rng.uniform(3, 7), rng.uniform(3, 7), rng.uniform(0, np.pi)
    h = relief * np.sin(fx * np.pi * u + ph) * np.sin(fy * np.pi * v)
    # flatten the relief toward the border so the box's hard edges stay straight
    w = np.clip(np.minimum(np.minimum(u, 1 - u), np.minimum(v, 1 - v)) / 0.1, 0, 1)
    h = h * w
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                        np.stack([a, c, d], -1).reshape(-1, 3)])
    return np.stack([u.ravel(), v.ravel()], 1), h.ravel(), F


def _vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    acc = np.zeros_like(V, dtype=np.float64)
    for k in range(3):
        np.add.at(acc, F[:, k], fn)
    return acc / (np.linalg.norm(acc, axis=1, keepdims=True) + 1e-20)


# the six faces of a unit cube: (origin, u axis, v axis) — u x v points OUTWARD
_CUBE = [
    ((-1, -1, 1), (1, 0, 0), (0, 1, 0)),     # +Z
    ((1, -1, -1), (-1, 0, 0), (0, 1, 0)),    # -Z
    ((1, -1, 1), (0, 0, -1), (0, 1, 0)),     # +X
    ((-1, -1, -1), (0, 0, 1), (0, 1, 0)),    # -X
    ((-1, 1, 1), (1, 0, 0), (0, 0, -1)),     # +Y
    ((-1, -1, -1), (1, 0, 0), (0, 0, 1)),    # -Y
]


def crate_sides(n=40, half=0.5, relief=0.02, tiling=False):
    """Six bumpy box sides, each its OWN set of vertices (hard edges: the normals split at every
    box edge, exactly like a game prop exported with smoothing groups). Each side gets a cell of a
    3x2 atlas, or — `tiling=True` — a world-scaled UV that runs far outside [0,1].
    Returns a list of (V, F, UV, N) per side, all in the same space."""
    sides = []
    for k, (o, du, dv) in enumerate(_CUBE):
        o, du, dv = (np.asarray(x, np.float64) for x in (o, du, dv))
        uv01, h, F = _grid(n, relief, seed=k)
        nrm = np.cross(du, dv)
        V = (o + 2 * uv01[:, :1] * du + 2 * uv01[:, 1:] * dv + h[:, None] * nrm) * half
        if tiling:
            UV = uv01 * 4.0 + np.array([k * 0.37, 0.0])          # repeats 4x, offset per side
        else:
            cell = np.array([k % 3, k // 3], np.float64)
            UV = (cell + 0.02 + uv01 * 0.96) / np.array([3.0, 2.0])
        N = _vertex_normals(V, F)
        # orient outward
        if (N.mean(0) @ nrm) < 0:
            F = F[:, ::-1]
            N = -N
        sides.append((V.astype(np.float32), F.astype(np.int64), UV.astype(np.float32),
                      N.astype(np.float32)))
    return sides


def floor_tile(n=120, size=4.0, relief=0.08):
    """A flat floor in the XZ plane (zero height in Y) with fine relief — the case a Y-extent-based
    cage collapses on."""
    uv01, h, F = _grid(n, relief, seed=11)
    V = np.stack([(uv01[:, 0] - 0.5) * size, h, (uv01[:, 1] - 0.5) * size], 1)
    F = F[:, ::-1]                                                 # +Y up
    N = _vertex_normals(V, F)
    return V.astype(np.float32), F.astype(np.int64), uv01.astype(np.float32), N.astype(np.float32)


def merge(parts):
    """Concatenate (V, F, UV, N) parts into one."""
    Vs, Fs, UVs, Ns, off = [], [], [], [], 0
    for V, F, UV, N in parts:
        Vs.append(V); Fs.append(F + off); UVs.append(UV); Ns.append(N); off += len(V)
    return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(UVs), np.concatenate(Ns)


def flat_normal_map(res=64):
    img = np.zeros((res, res, 3), np.uint8)
    img[..., 0] = img[..., 1] = 128
    img[..., 2] = 255
    return img


def write_scene(path, meshes, materials=None, images=None, nodes=None, share_position=False,
                extra_attrs=None):
    """Write a GLB.

    meshes:    list of meshes, each a list of prims {"V","F","UV","N","material"} (UV/N optional).
    materials: list of dicts {"normal": image index or None, "base": image index or None,
               "normal_texcoord": int}.
    images:    list of uint8 (H, W, 3) arrays, embedded as PNG.
    nodes:     list of (mesh index, translation) — defaults to one node per mesh at the origin.
    share_position: within each mesh, every prim indexes ONE shared POSITION/NORMAL/UV set (what
               several exporters do for multi-material meshes).
    extra_attrs: {name: (array per prim list, componentType, type, normalized)} written verbatim.
    """
    blob = bytearray()
    views, accs = [], []

    def add(arr, comp, gtype, minmax=False, normalized=False):
        data = np.ascontiguousarray(arr, dtype=_CT[comp])
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob.extend(data.tobytes())
        views.append(BufferView(buffer=0, byteOffset=off, byteLength=data.nbytes))
        a = Accessor(bufferView=len(views) - 1, byteOffset=0, componentType=comp,
                     count=len(data), type=gtype, normalized=normalized)
        if minmax:
            a.min = data.reshape(len(data), -1).min(0).tolist()
            a.max = data.reshape(len(data), -1).max(0).tolist()
        accs.append(a)
        return len(accs) - 1

    g = GLTF2()
    gl_meshes = []
    for mi, prims in enumerate(meshes):
        out = []
        if share_position:
            V, F_all, UV, N = merge([(p["V"], p["F"], p["UV"], p["N"]) for p in prims])
            pos = add(V, FLOAT, "VEC3", minmax=True)
            nrm = add(N, FLOAT, "VEC3")
            uv = add(UV, FLOAT, "VEC2")
            off = 0
            for p in prims:
                idx = add((p["F"] + off).reshape(-1), UNSIGNED_INT, "SCALAR")
                off += len(p["V"])
                out.append(Primitive(attributes=Attributes(POSITION=pos, NORMAL=nrm, TEXCOORD_0=uv),
                                     indices=idx, material=p.get("material")))
        else:
            for pi, p in enumerate(prims):
                at = Attributes(POSITION=add(p["V"], FLOAT, "VEC3", minmax=True))
                if p.get("N") is not None:
                    at.NORMAL = add(p["N"], FLOAT, "VEC3")
                if p.get("UV") is not None:
                    at.TEXCOORD_0 = add(p["UV"], FLOAT, "VEC2")
                for name, (arrs, comp, gtype, normd) in (extra_attrs or {}).items():
                    setattr(at, name, add(arrs[mi][pi], comp, gtype, normalized=normd))
                out.append(Primitive(attributes=at,
                                     indices=add(p["F"].reshape(-1), UNSIGNED_INT, "SCALAR"),
                                     material=p.get("material")))
        gl_meshes.append(Mesh(primitives=out))

    gl_images, textures = [], []
    for img in (images or []):
        buf = io.BytesIO()
        Image.fromarray(img, mode="RGB").save(buf, format="PNG")
        raw = buf.getvalue()
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob.extend(raw)
        views.append(BufferView(buffer=0, byteOffset=off, byteLength=len(raw)))
        gl_images.append(GLTFImage(bufferView=len(views) - 1, mimeType="image/png"))
        textures.append(Texture(source=len(gl_images) - 1))

    mats = []
    for m in (materials or [{}]):
        mat = Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0))
        if m.get("normal") is not None:
            mat.normalTexture = NormalMaterialTexture(index=m["normal"], scale=1.0,
                                                      texCoord=m.get("normal_texcoord", 0))
        if m.get("base") is not None:
            mat.pbrMetallicRoughness.baseColorTexture = TextureInfo(index=m["base"])
        mats.append(mat)

    g.meshes = gl_meshes
    g.materials = mats
    g.images = gl_images
    g.textures = textures
    nodes = nodes or [(i, (0.0, 0.0, 0.0)) for i in range(len(meshes))]
    g.nodes = [Node(mesh=m, translation=list(t)) for m, t in nodes]
    g.scenes = [Scene(nodes=list(range(len(g.nodes))))]
    g.scene = 0
    g.bufferViews = views
    g.accessors = accs
    g.buffers = [Buffer(byteLength=len(blob))]
    g.set_binary_blob(bytes(blob))
    g.save(path)
    return path


def read_prims(path):
    """[(mesh index, prim index, P, F, attrs dict, material)] straight from the file."""
    from retopotool.gltf_io import _acc
    g = GLTF2().load(path)
    blob = g.binary_blob()
    out = []
    for mi, mesh in enumerate(g.meshes or []):
        for pi, prim in enumerate(mesh.primitives):
            at = prim.attributes
            P = _acc(g, blob, at.POSITION).astype(np.float64)
            F = _acc(g, blob, prim.indices).astype(np.int64).reshape(-1, 3)
            attrs = {k: _acc(g, blob, v) for k, v in vars(at).items()
                     if v is not None and k != "POSITION"}
            out.append((mi, pi, P, F, attrs, prim.material))
    return g, out
