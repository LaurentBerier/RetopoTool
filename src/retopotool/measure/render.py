"""Textured orthographic render of a GLB, in numpy. No GPU, no browser, no Blender.

Written because mesh-processing defects are often ones only the RENDER shows — a warped decal, a
smeared iris — and a headless check should not need a browser or a GPU. A z-buffered barycentric
rasterizer over the base-colour texture is a hundred lines and answers the question directly.

It renders geometry + base colour + a fixed headlight, and nothing else: no normal map, no PBR, no
shadows. That is deliberate — it makes two renders of the same character comparable, which is what a
before/after is for. Do not read it as a preview of what the engine will show.

    retopo measure render CHARACTER.glb out.png
    retopo measure render CHARACTER.glb chest.png --frac 0.32 0.68 0.52 0.72   # crop
    retopo measure render CHARACTER.glb back.png --back
"""
from __future__ import annotations

import argparse
import io

import numpy as np


# Y is up and Z is front (glTF convention, character facing +Z), so a front view looks down -Z and
# the two screen axes are X and Y.
_VIEW_AXIS = 2
_LIGHT = np.array([0.30, 0.40, 0.86])


def load_textured(path: str):
    """Concatenated textured triangle geometry in world space + the first base-colour image found
    (a multi-material prop is drawn with that one texture — this is a before/after instrument)."""
    from PIL import Image
    from ..bake_normals import _texture_image_index_compat
    from ..gltf_io import concat_triangles, load_triangles
    g, blob, prims = load_triangles(path)
    prims = [p for p in prims if p["UV"] is not None]
    if not prims:
        raise ValueError(f"{path}: no textured triangle primitive")
    c = concat_triangles(prims, keys=("UV",))
    tex = np.ones((1, 1, 3)) * 0.8
    for p in prims:
        mat = g.materials[p["material"]] if (p["material"] is not None and g.materials) else None
        pbr = mat.pbrMetallicRoughness if mat is not None else None
        if pbr is None or pbr.baseColorTexture is None:
            continue
        ii = _texture_image_index_compat(g, pbr.baseColorTexture.index)
        img = g.images[ii] if (ii is not None and g.images and ii < len(g.images)) else None
        if img is None or img.bufferView is None:
            continue
        bv = g.bufferViews[img.bufferView]
        o = bv.byteOffset or 0
        im = Image.open(io.BytesIO(bytes(blob[o:o + bv.byteLength]))).convert("RGB")
        tex = np.asarray(im, dtype=np.float64) / 255.0
        break
    return c["P"], c["UV"], c["F"], tex


def _sample(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear texture fetch. glTF puts UV (0,0) at the image's TOP-left, so v is not flipped —
    getting that backwards on a fragmented atlas returns plausible-looking garbage, not an obvious
    upside-down image."""
    h, w, _ = tex.shape
    x = np.clip(uv[:, 0], 0, 1) * (w - 1)
    y = np.clip(uv[:, 1], 0, 1) * (h - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    return ((tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx) * (1 - fy)
            + (tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx) * fy)


def rasterize(P, UV, F, tex, box, res: int, front: bool = True) -> np.ndarray:
    ax = [i for i in range(3) if i != _VIEW_AXIS]
    ulo, uhi, vlo, vhi = box
    facing = 1.0 if front else -1.0
    zbuf = np.full((res, res), -np.inf)
    out = np.full((res, res, 3), 0.08)

    tri = P[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    keep = (n[:, _VIEW_AXIS] * facing) > 0
    Fk, tri, n = F[keep], tri[keep], n[keep]
    if not len(Fk):
        return (out * 255).astype(np.uint8)
    su = (tri[:, :, ax[0]] - ulo) / (uhi - ulo) * (res - 1)
    if not front:                       # mirror X so a back view reads left-right correctly
        su = (res - 1) - su
    sv = (vhi - tri[:, :, ax[1]]) / (vhi - vlo) * (res - 1)
    sz = tri[:, :, _VIEW_AXIS] * facing
    uvt = UV[Fk]
    lam = np.clip((n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-30))
                  @ (_LIGHT / np.linalg.norm(_LIGHT)) * facing, 0, 1) * 0.55 + 0.45

    lo_u = np.floor(su.min(1)).astype(int)
    hi_u = np.ceil(su.max(1)).astype(int)
    lo_v = np.floor(sv.min(1)).astype(int)
    hi_v = np.ceil(sv.max(1)).astype(int)
    for k in np.nonzero((hi_u >= 0) & (lo_u < res) & (hi_v >= 0) & (lo_v < res))[0]:
        x0, x1 = max(lo_u[k], 0), min(hi_u[k], res - 1)
        y0, y1 = max(lo_v[k], 0), min(hi_v[k], res - 1)
        if x1 < x0 or y1 < y0:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        ax0, ay0 = su[k, 0], sv[k, 0]
        bx, by = su[k, 1], sv[k, 1]
        cx, cy = su[k, 2], sv[k, 2]
        det = (by - cy) * (ax0 - cx) + (cx - bx) * (ay0 - cy)
        if abs(det) < 1e-12:
            continue
        w0 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / det
        w1 = ((cy - ay0) * (X - cx) + (ax0 - cx) * (Y - cy)) / det
        w2 = 1 - w0 - w1
        m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not m.any():
            continue
        z = (w0 * sz[k, 0] + w1 * sz[k, 1] + w2 * sz[k, 2])[m]
        Xi, Yi = X[m], Y[m]
        better = z > zbuf[Yi, Xi]
        if not better.any():
            continue
        Xi, Yi = Xi[better], Yi[better]
        b0, b1, b2 = w0[m][better][:, None], w1[m][better][:, None], w2[m][better][:, None]
        zbuf[Yi, Xi] = z[better]
        out[Yi, Xi] = _sample(tex, b0 * uvt[k, 0] + b1 * uvt[k, 1] + b2 * uvt[k, 2]) * lam[k]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def render_ortho(glb: str, out_png: str, frac=None, res: int = 700, front: bool = True) -> str:
    from PIL import Image
    P, UV, F, tex = load_textured(glb)
    ax = [i for i in range(3) if i != _VIEW_AXIS]
    lo, hi = P.min(0), P.max(0)
    if frac:
        box = (lo[ax[0]] + frac[0] * (hi[ax[0]] - lo[ax[0]]),
               lo[ax[0]] + frac[1] * (hi[ax[0]] - lo[ax[0]]),
               lo[ax[1]] + frac[2] * (hi[ax[1]] - lo[ax[1]]),
               lo[ax[1]] + frac[3] * (hi[ax[1]] - lo[ax[1]]))
    else:
        box = (lo[ax[0]], hi[ax[0]], lo[ax[1]], hi[ax[1]])
    Image.fromarray(rasterize(P, UV, F, tex, box, res, front)).save(out_png)
    return out_png


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="retopo measure render", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glb")
    ap.add_argument("out")
    ap.add_argument("--res", type=int, default=700)
    ap.add_argument("--frac", type=float, nargs=4, default=None,
                    metavar=("XLO", "XHI", "YLO", "YHI"))
    ap.add_argument("--back", action="store_true", help="view from behind")
    a = ap.parse_args(argv)
    print("wrote", render_ortho(a.glb, a.out, frac=a.frac, res=a.res, front=not a.back))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
