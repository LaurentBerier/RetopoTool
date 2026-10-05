"""Bake high-resolution surface detail into a tangent-space normal map for a decimated mesh.

This is the second half of the optimize / LOD step in `retopotool.decimate`. Decimation alone throws
away the 2-5 mm relief (eyelids, lips, wrinkles, seams, fabric folds) that makes a generated
character read as detailed; the AAA answer is to keep the LOW-poly mesh and put that relief back as
a normal map baked from the high-poly original. Nothing here changes the silhouette — it changes
what the surface looks like between the triangles.

Why it is cheap enough to run inline (all measured on a 3.0M-tri -> 401k-tri character):

    igl.AABB build over 3.0M triangles        3.1 s
    UV raster of 401k low-poly tris @2048     1.1 s
    2 x 2.8M cage rays (100% hit)             1.2 s

The decimator keeps the SOURCE UV LAYOUT exactly (see `retopotool.decimate`), which is what makes this a
bake and not a retopo: base colour and metallic-roughness stay valid untouched, paint edits survive,
and only the normal map is rewritten. A re-unwrap would invalidate all three.

Pipeline, per material:

  1. Rebuild the low-poly shading basis. Smooth normals come from the WELDED low-poly (so they cross
     UV seams), tangents are Lengyel-accumulated per split vertex from the low-poly UVs (so they do
     NOT cross seams). Both are written back onto the mesh, because a normal map is only meaningful
     in the exact tangent frame it was baked in — the renderer must use this basis, not one it
     derives itself.
  2. Rasterize the low-poly triangles into UV space at the bake resolution, giving each texel its
     triangle and barycentric coordinates.
  3. Cast a cage ray from each covered texel along +/- the interpolated low-poly normal and keep the
     nearest hit on the high-poly whose normal agrees (a plain nearest-hit picks up the far side of
     a limb across a gap: the raw hit distribution has a 0.04 mm median and a 25 mm p99).
  4. Compose. Sample the SOURCE normal map at the hit's UV, rebuild that vector in the high-poly's
     tangent frame, and re-express the result in the low-poly's frame. Composing rather than
     overwriting is the point: generators like Meshy ship a 2K-4K normal map of micro-detail that no geometric bake
     can reproduce, so the output carries both the micro-detail AND the geometry the decimation
     removed.
  5. Dilate the result past the island borders so bilinear filtering and mip generation never pull
     background in across a UV seam.

Ray queries go through libigl (`igl.AABB`), which is a pinned dependency. trimesh's ray/proximity
subsystems need the optional `rtree` package (and raise `ModuleNotFoundError: rtree` without it) —
do not reach for them.
"""
from __future__ import annotations

import io
import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Bake resolution follows the source normal map, clamped to this band. Below 2048 a face occupies
# too few texels to be worth the pass; above 4096 the GLB grows faster than the detail does.
BAKE_MIN_RES = 2048
BAKE_MAX_RES = 4096
# A caller that is deliberately building a small asset (an LOD for mobile) may ask for less than
# BAKE_MIN_RES — there the map is the whole point and a 512 or 1024 map is the shipping size. This
# is the floor below which the bake stops being worth its runtime at all.
BAKE_HARD_MIN = 256
# Cage half-height as a fraction of the model's bbox height. The low-poly is a vertex SUBSET of the
# high-poly, so the two surfaces are never far apart; 1% of height (~1.8 cm on a human) covers the
# deepest decimated-away cavity while staying well under the gap between an arm and the torso.
CAGE_FRAC = 0.01
# A hit only counts if the high-poly normal there agrees with the low-poly normal. Without it a ray
# leaving a thin surface can register on the far side of a limb.
NORMAL_AGREE = 0.0
# Texels of padding pushed outward from each UV island.
DILATE_PX = 16
# Smallest tangent-space z we will encode. A hit whose normal tips past the low-poly tangent plane
# would otherwise encode a normal pointing INTO the surface, which every renderer shades as black.
MIN_Z = 0.05


def _smooth_normals(P: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals computed on the POSITION-WELDED mesh, scattered back to the
    split vertices. Welding matters: computed per split vertex, every UV seam would shade as a
    hard crease."""
    tol = max(float(np.linalg.norm(P.max(0) - P.min(0))), 1e-6) * 1e-6
    _, weld = np.unique(np.round(P.astype(np.float64) / tol).astype(np.int64),
                        axis=0, return_inverse=True)
    Fw = weld[F]
    e1 = P[F[:, 1]] - P[F[:, 0]]
    e2 = P[F[:, 2]] - P[F[:, 0]]
    fn = np.cross(e1, e2)                      # length = 2 * area -> area weighting for free
    acc = np.zeros((weld.max() + 1, 3), dtype=np.float64)
    for k in range(3):
        np.add.at(acc, Fw[:, k], fn)
    ln = np.linalg.norm(acc, axis=1, keepdims=True)
    # A welded vertex whose incident face normals cancel exactly (a zero-area sliver, a pinched
    # fold) accumulates to zero. Never ship that: a zero NORMAL shades black and makes the tangent
    # frame — and therefore every baked texel around it — undefined. Retry those with UNWEIGHTED
    # face normals, which cannot cancel for a fan that is not perfectly degenerate.
    degenerate = (ln <= 1e-20).ravel()
    if degenerate.any():
        fl = np.linalg.norm(fn, axis=1, keepdims=True)
        fnn = np.divide(fn, fl, out=np.zeros_like(fn), where=fl > 1e-20)
        acc2 = np.zeros_like(acc)
        for k in range(3):
            np.add.at(acc2, Fw[:, k], fnn)
        acc[degenerate] = acc2[degenerate]
        ln = np.linalg.norm(acc, axis=1, keepdims=True)
        still = (ln <= 1e-20).ravel()
        if still.any():
            # Last resort: point away from the mesh centroid. Arbitrary, but finite and outward,
            # which is the right sign for a closed character surface.
            centre = P.mean(0)
            idx = np.nonzero(still)[0]
            rep = np.zeros((len(acc), 3))     # representative position per welded slot
            rep[weld] = P.astype(np.float64)
            d = rep[idx] - centre
            dl = np.linalg.norm(d, axis=1, keepdims=True)
            acc[idx] = np.divide(d, dl, out=np.tile([0.0, 1.0, 0.0], (len(idx), 1)), where=dl > 1e-20)
            ln = np.linalg.norm(acc, axis=1, keepdims=True)
    n = acc[weld]
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, ln, out=np.tile(np.array([0.0, 1.0, 0.0]), (len(n), 1)),
                     where=ln > 1e-20)


def _tangents(P: np.ndarray, F: np.ndarray, UV: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Per-split-vertex tangents (Lengyel), Gram-Schmidt orthogonalized against N, with the
    bitangent handedness in w. NOT welded — a tangent is a function of the UV parameterization, so
    the two sides of a seam legitimately disagree and averaging them twists the frame."""
    e1 = P[F[:, 1]] - P[F[:, 0]]
    e2 = P[F[:, 2]] - P[F[:, 0]]
    d1 = UV[F[:, 1]] - UV[F[:, 0]]
    d2 = UV[F[:, 2]] - UV[F[:, 0]]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.divide(1.0, det, out=np.zeros_like(det), where=np.abs(det) > 1e-20)
    tan = (e1 * d2[:, 1, None] - e2 * d1[:, 1, None]) * r[:, None]
    bit = (e2 * d1[:, 0, None] - e1 * d2[:, 0, None]) * r[:, None]
    Tacc = np.zeros_like(P, dtype=np.float64)
    Bacc = np.zeros_like(P, dtype=np.float64)
    for k in range(3):
        np.add.at(Tacc, F[:, k], tan)
        np.add.at(Bacc, F[:, k], bit)
    T = Tacc - N * (N * Tacc).sum(1, keepdims=True)          # Gram-Schmidt
    ln = np.linalg.norm(T, axis=1, keepdims=True)
    degenerate = (ln <= 1e-12).ravel()
    if degenerate.any():
        # Any vector perpendicular to N will do where the UVs give no direction (a fully collapsed
        # UV triangle); an arbitrary-but-stable choice beats a NaN.
        fallback = np.tile(np.array([1.0, 0.0, 0.0]), (degenerate.sum(), 1))
        alt = np.abs(N[degenerate, 0]) > 0.9
        fallback[alt] = np.array([0.0, 1.0, 0.0])
        f = fallback - N[degenerate] * (N[degenerate] * fallback).sum(1, keepdims=True)
        T[degenerate] = f
        ln = np.linalg.norm(T, axis=1, keepdims=True)
    T = np.divide(T, ln, out=np.zeros_like(T), where=ln > 1e-20)
    w = np.where((np.cross(N, T) * Bacc).sum(1) < 0.0, -1.0, 1.0)
    return np.concatenate([T, w[:, None]], axis=1)


def _rasterize_uv(UV: np.ndarray, F: np.ndarray, res: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-texel (triangle id, barycentric) for the UV-space triangles. -1 where uncovered.

    Triangles are batched by UV bounding-box size class so one vectorized pass can cover a whole
    class without allocating the largest triangle's box for every triangle (a character atlas has a
    7-texel median box and a 383-texel maximum)."""
    tri_id = np.full((res, res), -1, dtype=np.int64)
    bary = np.zeros((res, res, 3), dtype=np.float32)
    uv = UV[F] * res                                       # (T, 3, 2) in texel units
    mn = np.floor(uv.min(1)).astype(np.int64)
    span = np.ceil(uv.max(1)).astype(np.int64) - mn
    done = np.zeros(len(F), dtype=bool)
    k = 2
    while not done.all():
        sel = np.where((~done) & (span.max(1) <= k))[0]
        done[sel] = True
        if len(sel):
            batch = max(1, 4_000_000 // (k * k))
            for s in range(0, len(sel), batch):
                idx = sel[s:s + batch]
                a, b, c = uv[idx, 0], uv[idx, 1], uv[idx, 2]
                ox = mn[idx, 0][:, None] + np.arange(k)[None, :]
                oy = mn[idx, 1][:, None] + np.arange(k)[None, :]
                px = ox[:, :, None] + 0.5
                py = oy[:, None, :] + 0.5
                det = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                       - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))[:, None, None]
                det = np.where(np.abs(det) < 1e-12, 1e-12, det)
                la = ((b[:, 0][:, None, None] - px) * (c[:, 1][:, None, None] - py)
                      - (c[:, 0][:, None, None] - px) * (b[:, 1][:, None, None] - py)) / det
                lb = ((c[:, 0][:, None, None] - px) * (a[:, 1][:, None, None] - py)
                      - (a[:, 0][:, None, None] - px) * (c[:, 1][:, None, None] - py)) / det
                lc = 1.0 - la - lb
                inside = (la >= -1e-4) & (lb >= -1e-4) & (lc >= -1e-4)
                ii, jj, kk = np.nonzero(inside)
                if not len(ii):
                    continue
                X = ox[ii, jj]
                Y = oy[ii, kk]
                ok = (X >= 0) & (X < res) & (Y >= 0) & (Y < res)
                tri_id[Y[ok], X[ok]] = idx[ii[ok]]
                bary[Y[ok], X[ok]] = np.stack([la[ii, jj, kk], lb[ii, jj, kk],
                                               lc[ii, jj, kk]], axis=1)[ok]
        if k >= res:
            break
        k *= 2
    return tri_id, bary


def _sample_bilinear(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sample of an (H, W, C) float image at UV in [0,1], V pointing down (glTF)."""
    h, w = img.shape[:2]
    x = np.clip(uv[:, 0] * w - 0.5, 0, w - 1)
    y = np.clip(uv[:, 1] * h - 0.5, 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    return ((img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy)
            + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy)


def _dilate(rgb: np.ndarray, mask: np.ndarray, iters: int = DILATE_PX) -> np.ndarray:
    """Push covered texels outward into the background so filtering/mipping never samples across an
    island border. Each pass averages the covered neighbours of an uncovered texel."""
    out = rgb.astype(np.float32).copy()
    have = mask.copy()
    for _ in range(iters):
        if have.all():
            break
        acc = np.zeros_like(out)
        cnt = np.zeros(have.shape, dtype=np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            src_v = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
            src_m = np.roll(np.roll(have, dy, axis=0), dx, axis=1)
            acc += src_v * src_m[..., None]
            cnt += src_m
        grow = (~have) & (cnt > 0)
        out[grow] = acc[grow] / cnt[grow][:, None]
        have |= grow
    return out


def bake_normal_map(
    P_hi: np.ndarray, F_hi: np.ndarray, UV_hi: np.ndarray,
    N_hi: Optional[np.ndarray], TAN_hi: Optional[np.ndarray],
    src_normal_map: Optional[np.ndarray],
    P_lo: np.ndarray, F_lo: np.ndarray, UV_lo: np.ndarray,
    res: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Bake `P_hi`'s surface detail into a tangent-space normal map for `P_lo`.

    Returns `(rgb uint8 (res, res, 3), low-poly NORMAL (n, 3), low-poly TANGENT (n, 4), stats)`.
    The returned normal/tangent MUST be written onto the low-poly primitive: the map is only valid
    in the basis it was baked against.
    """
    import igl

    N_lo = _smooth_normals(P_lo, F_lo)
    T4_lo = _tangents(P_lo, F_lo, UV_lo, N_lo)
    T_lo = T4_lo[:, :3]
    B_lo = np.cross(N_lo, T_lo) * T4_lo[:, 3:4]

    if N_hi is None:
        N_hi = _smooth_normals(P_hi, F_hi)
    else:
        N_hi = N_hi.astype(np.float64)
        ln = np.linalg.norm(N_hi, axis=1, keepdims=True)
        N_hi = np.divide(N_hi, ln, out=np.zeros_like(N_hi), where=ln > 1e-20)
    if src_normal_map is not None:
        if TAN_hi is None:
            TAN_hi = _tangents(P_hi, F_hi, UV_hi, N_hi)
        T_hi = TAN_hi[:, :3].astype(np.float64)
        B_hi = np.cross(N_hi, T_hi) * TAN_hi[:, 3:4].astype(np.float64)

    tri_id, bary = _rasterize_uv(UV_lo, F_lo, res)
    ys, xs = np.nonzero(tri_id >= 0)
    if not len(ys):
        raise ValueError("UV rasterization covered no texels — the mesh has no usable UV layout")
    tid = tri_id[ys, xs]
    bw = bary[ys, xs].astype(np.float64)
    corners = F_lo[tid]

    def lerp_lo(A):
        return (A[corners] * bw[:, :, None]).sum(1)

    O = lerp_lo(P_lo)
    Nt = lerp_lo(N_lo)
    Nt /= np.linalg.norm(Nt, axis=1, keepdims=True) + 1e-20
    Tt = lerp_lo(T_lo)
    Tt -= Nt * (Nt * Tt).sum(1, keepdims=True)
    Tt /= np.linalg.norm(Tt, axis=1, keepdims=True) + 1e-20
    Bt = lerp_lo(B_lo)
    Bt -= Nt * (Nt * Bt).sum(1, keepdims=True) + Tt * (Tt * Bt).sum(1, keepdims=True)
    Bt /= np.linalg.norm(Bt, axis=1, keepdims=True) + 1e-20

    eps = max(CAGE_FRAC * float(P_hi[:, 1].max() - P_hi[:, 1].min()), 1e-5)
    tree = igl.AABB()
    tree.init(P_hi, F_hi)
    Fh64 = F_hi.astype(np.int64)
    # Offsetting the origin and searching forward avoids the t=0 self-hit where the low-poly and
    # high-poly surfaces touch (every low-poly vertex IS a high-poly vertex).
    f_f, t_f, b_f = tree.intersect_ray_first(P_hi, Fh64, O - Nt * eps, Nt)
    f_b, t_b, b_b = tree.intersect_ray_first(P_hi, Fh64, O + Nt * eps, -Nt)

    def resolve(fid, t, b2, sign):
        fid = np.asarray(fid, dtype=np.int64)
        t = np.asarray(t, dtype=np.float64)
        b2 = np.asarray(b2, dtype=np.float64)
        s = np.where(np.isfinite(t), sign * (t - eps), np.inf)
        hit = (fid >= 0) & np.isfinite(t) & (np.abs(s) <= eps)
        # igl leaves the barycentric row UNINITIALIZED on a miss — it can hold inf/nan or a value
        # large enough that `1 - b1 - b2` overflows. Zero the misses before any arithmetic; their
        # weights are discarded by `hit` anyway, but a nan here poisons the whole vectorized sum.
        b2 = np.where(hit[:, None] & np.isfinite(b2), b2, 0.0)
        # Barycentric convention: igl returns (b1, b2) for v1, v2; v0 takes the remainder.
        w = np.stack([1.0 - b2[:, 0] - b2[:, 1], b2[:, 0], b2[:, 1]], axis=1)
        cor = np.where(hit[:, None], Fh64[np.clip(fid, 0, None)], 0)
        n = (N_hi[cor] * w[:, :, None]).sum(1)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-20
        hit &= (n * Nt).sum(1) > NORMAL_AGREE
        return hit, np.where(np.isfinite(s), np.abs(s), np.inf), s, cor, w

    hit_f, d_f, s_f, cor_f, w_f = resolve(f_f, t_f, b_f, 1.0)
    hit_b, d_b, s_b, cor_b, w_b = resolve(f_b, t_b, b_b, -1.0)
    take_f = hit_f & (~hit_b | (d_f <= d_b))
    hit = hit_f | hit_b
    cor = np.where(take_f[:, None], cor_f, cor_b)
    wgt = np.where(take_f[:, None], w_f, w_b)

    if (~hit).any():
        # Rays that found nothing inside the cage (grazing a decimated-away crease, a hole in the
        # high-poly). Fall back to the closest point on the high-poly, which always exists.
        miss = np.nonzero(~hit)[0]
        _, fid_c, pts_c = igl.point_mesh_squared_distance(O[miss], P_hi, Fh64)
        fid_c = np.asarray(fid_c, dtype=np.int64)
        tri = P_hi[Fh64[fid_c]]
        # Barycentric of the returned closest point, via the triangle's own normal-projected areas.
        v0 = tri[:, 1] - tri[:, 0]
        v1 = tri[:, 2] - tri[:, 0]
        v2 = np.asarray(pts_c) - tri[:, 0]
        d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
        d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
        den = d00 * d11 - d01 * d01
        den = np.where(np.abs(den) < 1e-20, 1e-20, den)
        v = (d11 * d20 - d01 * d21) / den
        w2 = (d00 * d21 - d01 * d20) / den
        cor[miss] = Fh64[fid_c]
        wgt[miss] = np.stack([1.0 - v - w2, v, w2], axis=1)

    # --- compose: high-poly geometric normal, modulated by the source normal map ---
    n_world = (N_hi[cor] * wgt[:, :, None]).sum(1)
    n_world /= np.linalg.norm(n_world, axis=1, keepdims=True) + 1e-20
    if src_normal_map is not None:
        uv_hit = (UV_hi[cor] * wgt[:, :, None]).sum(1)
        ts = _sample_bilinear(src_normal_map, uv_hit) * 2.0 - 1.0
        th = (T_hi[cor] * wgt[:, :, None]).sum(1)
        bh = (B_hi[cor] * wgt[:, :, None]).sum(1)
        n_world = (th * ts[:, 0:1] + bh * ts[:, 1:2] + n_world * ts[:, 2:3])
        n_world /= np.linalg.norm(n_world, axis=1, keepdims=True) + 1e-20

    nx = (n_world * Tt).sum(1)
    ny = (n_world * Bt).sum(1)
    nz = (n_world * Nt).sum(1)
    # Never encode a normal that leans past the low-poly tangent plane (renders black).
    below = nz < MIN_Z
    if below.any():
        planar = np.sqrt(np.maximum(1.0 - MIN_Z * MIN_Z, 0.0))
        ln = np.sqrt(nx[below] ** 2 + ny[below] ** 2) + 1e-20
        nx[below] = nx[below] / ln * planar
        ny[below] = ny[below] / ln * planar
        nz[below] = MIN_Z
    ts_out = np.stack([nx, ny, nz], axis=1)
    ts_out /= np.linalg.norm(ts_out, axis=1, keepdims=True) + 1e-20

    rgb = np.zeros((res, res, 3), dtype=np.float32)
    rgb[:, :, 2] = 1.0                                  # flat normal where nothing was baked
    rgb[:, :, 0] = 0.5
    rgb[:, :, 1] = 0.5
    rgb[ys, xs] = (ts_out * 0.5 + 0.5).astype(np.float32)
    covered = np.zeros((res, res), dtype=bool)
    covered[ys, xs] = True
    rgb = _dilate(rgb, covered)
    out = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)

    stats = {
        "resolution": int(res),
        "texels_covered": int(covered.sum()),
        "coverage": float(covered.mean()),
        "cage_hit_rate": float(hit.mean()),
        "composed_with_source_map": src_normal_map is not None,
    }
    return out, N_lo.astype(np.float32), T4_lo.astype(np.float32), stats


def encode_png(rgb: np.ndarray) -> bytes:
    """PNG, always. A normal map is not a photograph — JPEG's chroma subsampling puts visible
    blocking into the X/Y channels, which is exactly the detail this pass exists to carry."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


def decode_image(raw: bytes) -> Optional[np.ndarray]:
    """Decode embedded image bytes to a float (H, W, 3) array in 0..1, or None if unreadable."""
    from PIL import Image
    try:
        with Image.open(io.BytesIO(raw)) as im:
            return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        logger.warning("retopotool: could not decode source image; baking geometry only")
        return None


def pick_resolution(src_sizes, allow_small: bool = False) -> int:
    """Bake at the source normal map's resolution, clamped to the supported band and rounded to a
    power of two (mip-friendly, and every consumer of this GLB expects POT character atlases).

    `allow_small` lowers the floor to `BAKE_HARD_MIN` for a caller that is deliberately targeting a
    small asset — without it an LOD asking for a 512 map would silently get 2048, which is larger
    than the geometry it is attached to.
    """
    best = max([s for s in src_sizes if s], default=0)
    lo = BAKE_HARD_MIN if allow_small else BAKE_MIN_RES
    res = int(np.clip(best or lo, lo, BAKE_MAX_RES))
    return int(2 ** int(round(np.log2(res))))


def _texture_image_index_compat(g, tex_index):
    """Image index behind a texture, core slot or EXT_texture_webp. Shared with the measurement
    tool so an analysis and the bake never disagree about which image a material points at."""
    if tex_index is None or not g.textures or tex_index >= len(g.textures):
        return None
    tex = g.textures[tex_index]
    if tex.source is not None:
        return tex.source
    src = ((tex.extensions or {}).get("EXT_texture_webp") or {}).get("source")
    return int(src) if src is not None else None
