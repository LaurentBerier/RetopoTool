"""Synthetic GLB builders shared by the test modules (no asset files needed)."""
import io

import numpy as np
from PIL import Image
from pygltflib import (GLTF2, Accessor, BufferView, Buffer, Mesh, Primitive, Attributes, Node,
                       Scene, Material, PbrMetallicRoughness, Image as GLTFImage, Texture,
                       NormalMaterialTexture, Skin, FLOAT, UNSIGNED_BYTE, UNSIGNED_INT)

from retopotool.gltf_io import _CT


# --------------------------------------------------------------------------------------------
# a bumpy UV sphere with a "head": fine relief at the top, coarse relief on the body, so a
# decimator that allocates uniformly and one that allocates by importance give different answers.
# --------------------------------------------------------------------------------------------
def _bumpy_sphere(n_theta=190, n_phi=190, height=1.8):
    th = np.linspace(0.001, np.pi - 0.001, n_theta)
    ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    # fine ripples near the top (theta small), coarse ones elsewhere
    fine = 0.012 * np.sin(26 * T) * np.sin(26 * P)
    coarse = 0.02 * np.sin(5 * T) * np.sin(5 * P)
    w = np.clip((np.cos(T) - 0.4) / 0.6, 0, 1)
    r = 1.0 + w * fine + (1 - w) * coarse
    X = r * np.sin(T) * np.cos(P)
    Y = r * np.cos(T)
    Z = r * np.sin(T) * np.sin(P)
    V = np.stack([X, Y, Z], axis=-1).reshape(-1, 3) * (height / 2.0)
    UV = np.stack([P / (2 * np.pi), T / np.pi], axis=-1).reshape(-1, 2)
    idx = np.arange(n_theta * n_phi).reshape(n_theta, n_phi)
    a = idx[:-1, :-1]; b = idx[:-1, 1:]; c = idx[1:, 1:]; d = idx[1:, :-1]
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                        np.stack([a, c, d], -1).reshape(-1, 3)])
    # TRUE geometric normals, not the analytic sphere normal: on this bumpy surface the two differ
    # by 5.2 deg median / 82 deg max, and using the analytic one would make every shading assertion
    # compare against a surface that is not the one in the file.
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    fn = np.cross(e1, e2)
    acc = np.zeros_like(V, dtype=np.float64)
    for k in range(3):
        np.add.at(acc, F[:, k].astype(np.int64), fn)
    N = acc / (np.linalg.norm(acc, axis=1, keepdims=True) + 1e-20)
    return V.astype(np.float32), F.astype(np.uint32), UV.astype(np.float32), N.astype(np.float32)


def _write_glb(path, V, F, UV, N, normal_map=None, scale=1.0):
    V = (V * scale).astype(np.float32)
    blob = bytearray()
    views, accs = [], []

    def add(arr, comp, gtype, minmax=False):
        data = np.ascontiguousarray(arr, dtype=_CT[comp])
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob.extend(data.tobytes())
        views.append(BufferView(buffer=0, byteOffset=off, byteLength=data.nbytes))
        a = Accessor(bufferView=len(views) - 1, byteOffset=0, componentType=comp,
                     count=len(data), type=gtype)
        if minmax:
            a.min = data.min(0).tolist()
            a.max = data.max(0).tolist()
        accs.append(a)
        return len(accs) - 1

    pos = add(V, FLOAT, "VEC3", minmax=True)
    nrm = add(N, FLOAT, "VEC3")
    uv = add(UV, FLOAT, "VEC2")
    idx = add(F.reshape(-1), UNSIGNED_INT, "SCALAR")

    g = GLTF2()
    mat = Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0))
    images, textures = [], []
    if normal_map is not None:
        buf = io.BytesIO()
        Image.fromarray(normal_map, mode="RGB").save(buf, format="PNG")
        raw = buf.getvalue()
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob.extend(raw)
        views.append(BufferView(buffer=0, byteOffset=off, byteLength=len(raw)))
        images.append(GLTFImage(bufferView=len(views) - 1, mimeType="image/png", name="normal"))
        textures.append(Texture(source=0))
        mat.normalTexture = NormalMaterialTexture(index=0, scale=1.0)
    g.materials = [mat]
    g.images = images
    g.textures = textures
    g.meshes = [Mesh(primitives=[Primitive(
        attributes=Attributes(POSITION=pos, NORMAL=nrm, TEXCOORD_0=uv), indices=idx, material=0)])]
    g.nodes = [Node(mesh=0)]
    g.scenes = [Scene(nodes=[0])]
    g.scene = 0
    g.bufferViews = views
    g.accessors = accs
    g.buffers = [Buffer(byteLength=len(blob))]
    g.set_binary_blob(bytes(blob))
    g.save(path)
    return path


def _write_skinned_glb(path, V, F, UV, N, n_joints=6, orphans=0):
    """A skinned version of the test sphere: `n_joints` bones stacked up Y, each vertex bound to the
    two nearest with a smooth blend. `orphans` appends unreferenced accessors, reproducing what rig
    writers that append rather than repack leave behind (the pre-rig POSITION/NORMAL/TANGENT —
    9 MB on a real character)."""
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
            a.min = data.min(0).tolist()
            a.max = data.max(0).tolist()
        accs.append(a)
        return len(accs) - 1

    lo, hi = V[:, 1].min(), V[:, 1].max()
    t = np.clip((V[:, 1] - lo) / max(hi - lo, 1e-9), 0, 1) * (n_joints - 1)
    j0 = np.floor(t).astype(np.int64).clip(0, n_joints - 1)
    j1 = np.minimum(j0 + 1, n_joints - 1)
    w1 = (t - j0).astype(np.float32)
    J = np.zeros((len(V), 4), dtype=np.uint8)
    W = np.zeros((len(V), 4), dtype=np.float32)
    J[:, 0], J[:, 1] = j0, j1
    W[:, 0], W[:, 1] = 1.0 - w1, w1
    # a bone that appears ONLY in the high index exercises the "joint index survives" assertion
    J[t > n_joints - 1.5, 0] = n_joints - 1

    pos = add(V, FLOAT, "VEC3", minmax=True)
    nrm = add(N, FLOAT, "VEC3")
    uv = add(UV, FLOAT, "VEC2")
    jnt = add(J, UNSIGNED_BYTE, "VEC4")
    wgt = add(W, FLOAT, "VEC4")
    idx = add(F.reshape(-1), UNSIGNED_INT, "SCALAR")
    ibm = add(np.tile(np.eye(4, dtype=np.float32).reshape(1, 16), (n_joints, 1)), FLOAT, "MAT4")
    for _ in range(orphans):
        add(V, FLOAT, "VEC3")                      # referenced by nothing

    g = GLTF2()
    g.materials = [Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0))]
    g.meshes = [Mesh(primitives=[Primitive(
        attributes=Attributes(POSITION=pos, NORMAL=nrm, TEXCOORD_0=uv, JOINTS_0=jnt, WEIGHTS_0=wgt),
        indices=idx, material=0)])]
    g.nodes = [Node(mesh=0, skin=0)] + [Node(name=f"bone_{i}") for i in range(n_joints)]
    g.skins = [Skin(joints=list(range(1, n_joints + 1)), inverseBindMatrices=ibm, skeleton=1)]
    g.scenes = [Scene(nodes=[0] + list(range(1, n_joints + 1)))]
    g.scene = 0
    g.bufferViews = views
    g.accessors = accs
    g.buffers = [Buffer(byteLength=len(blob))]
    g.set_binary_blob(bytes(blob))
    g.save(path)
    return path
