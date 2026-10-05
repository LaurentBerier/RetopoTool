"""How far a decimated mesh's texture has SLID against its source, in texels.

The complaint this exists for: after optimizing, the silhouette is perfect and the decal on the
chest is unreadable. No geometric metric sees that — point-to-surface error was already sub-millimetre
when the lettering turned to mush — because the defect lives in the parameterization, not the surface.

The metric: for every vertex of the decimated mesh, find where it sits on the SOURCE surface and read
the source UV there; the drift is the distance from that UV to the one the vertex actually carries,
scaled by the base-colour texture's resolution. A drift of N texels means the texture is painted N
texels off at that point, and N is directly comparable across meshes because it is expressed in the
units the artwork is authored in.

Identifying WHICH atlas chart to compare against matters more than the projection: a generated atlas has
thousands of small islands, and the geometrically nearest source triangle is routinely on a different
one (the two sides of a seam are the same point in space). The island is therefore taken from the
source vertex whose UV is nearest the emitted UV among the geometrically nearest few — which is exact
for both transfer schemes, since both derive a wedge's UV from within its own island.

    retopo measure uv-drift SOURCE.glb DECIMATED.glb [DECIMATED2.glb ...]
    retopo measure uv-drift SOURCE.glb DECIMATED.glb --render out.png --frac 0.32 0.68 0.52 0.72

`--render` writes a textured orthographic front view (no GPU, no browser) so the number can be
checked against the thing a person actually complains about. Run it on the source too and put the
two side by side: this pipeline has been burned before by offline metrics that improved while the
render got worse.
"""
from __future__ import annotations

import argparse
import io
import os

import numpy as np

from ..decimate import _closest_on_triangles, _gather_incident, _incident_faces, _uv_islands


def read_mesh(path: str) -> dict:
    """Every textured triangle primitive in world space, concatenated, plus the decoded
    base-colour image of the first one that has it (its size is the texel unit)."""
    from ..gltf_io import concat_triangles, load_triangles
    g, blob, prims = load_triangles(path)
    prims = [p for p in prims if p["UV"] is not None]
    if not prims:
        raise ValueError(f"{path}: no textured triangle primitive")
    c = concat_triangles(prims, keys=("UV",))
    out = {"P": c["P"], "UV": c["UV"], "F": c["F"], "tex": None, "tex_size": 2048}
    from ..bake_normals import _texture_image_index_compat
    for p in prims:
        mat = g.materials[p["material"]] if (p["material"] is not None and g.materials) else None
        pbr = mat.pbrMetallicRoughness if mat is not None else None
        if pbr is None or pbr.baseColorTexture is None:
            continue
        ii = _texture_image_index_compat(g, pbr.baseColorTexture.index)
        img = g.images[ii] if (ii is not None and g.images and ii < len(g.images)) else None
        if img is None or img.bufferView is None:
            continue
        from PIL import Image
        bv = g.bufferViews[img.bufferView]
        off = bv.byteOffset or 0
        im = Image.open(io.BytesIO(bytes(blob[off:off + bv.byteLength]))).convert("RGB")
        out["tex"] = np.asarray(im, dtype=np.float64) / 255.0
        out["tex_size"] = max(im.size)
        break
    return out


def uv_drift(src: dict, lo: dict, k: int = 6) -> np.ndarray:
    """Per-vertex UV drift of `lo` against `src`, in UV units."""
    from scipy.spatial import cKDTree

    P, F, UV = src["P"], src["F"], src["UV"]
    isl = _uv_islands(len(P), F)
    inc_faces, inc_start = _incident_faces(len(P), F)
    face_isl = isl[F[:, 0]]

    A = P[F[:, 0]]
    E1 = P[F[:, 1]] - A
    E2 = P[F[:, 2]] - A
    d00 = (E1 * E1).sum(1)
    d01 = (E1 * E2).sum(1)
    d11 = (E2 * E2).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)

    Q, QUV = lo["P"], lo["UV"]
    _, nn = cKDTree(P).query(Q, k=k, workers=-1)
    nn = np.atleast_2d(nn.reshape(len(Q), -1))
    # the chart the vertex claims: nearest source vertex IN UV among the nearest few in space
    duv_v = np.linalg.norm(UV[nn] - QUV[:, None, :], axis=2)
    claim = isl[nn[np.arange(len(Q)), np.argmin(duv_v, axis=1)]]

    drift = np.zeros(len(Q))
    CH = 20_000
    for s in range(0, len(Q), CH):
        e = min(s + CH, len(Q))
        near = nn[s:e].ravel()
        cand, ok = _gather_incident(inc_faces, inc_start, near)
        cand = cand.reshape(e - s, -1)
        ok = ok.reshape(e - s, -1)
        cs = np.where(ok, cand, 0).astype(np.int64)
        ok &= face_isl[cs] == claim[s:e][:, None]
        # the production closest-point routine, so the oracle and the thing it grades agree
        d3, bary = _closest_on_triangles(Q[s:e, None, :], A[cs], E1[cs], E2[cs],
                                         d00[cs], d01[cs], d11[cs], den[cs])
        d3 = np.where(ok, d3, np.inf)
        best = np.argmin(d3, axis=1)
        rows = np.arange(e - s)
        fb = cs[rows, best]
        b = bary[rows, best]
        true_uv = (b[:, 0:1] * UV[F[fb, 0]] + b[:, 1:2] * UV[F[fb, 1]] + b[:, 2:3] * UV[F[fb, 2]])
        d = np.linalg.norm(true_uv - QUV[s:e], axis=1)
        drift[s:e] = np.where(np.isfinite(d3[rows, best]), d, 0.0)
    return drift


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="retopo measure uv-drift", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("decimated", nargs="+")
    ap.add_argument("--render", help="write a textured orthographic view of each mesh next to it")
    ap.add_argument("--frac", type=float, nargs=4, default=None,
                    metavar=("XLO", "XHI", "YLO", "YHI"),
                    help="crop as bbox fractions (default: whole model)")
    ap.add_argument("--res", type=int, default=700)
    a = ap.parse_args(argv)

    src = read_mesh(a.source)
    print(f"source     {os.path.basename(a.source):48s} "
          f"{len(src['F']):>9,} tris  base colour {src['tex_size']}px")
    for path in a.decimated:
        lo = read_mesh(path)
        d = uv_drift(src, lo) * src["tex_size"]
        print(f"  {os.path.basename(path):48s} {len(lo['F']):>9,} tris  "
              f"drift texels p50 {np.median(d):6.2f}  p95 {np.percentile(d, 95):7.2f}  "
              f"max {d.max():8.1f}  >1 texel {100 * (d > 1).mean():5.1f}%")
    if a.render:
        from .render import render_ortho
        for path in [a.source] + list(a.decimated):
            out = f"{os.path.splitext(a.render)[0]}_{os.path.splitext(os.path.basename(path))[0]}.png"
            render_ortho(path, out, frac=a.frac, res=a.res)
            print("  wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
