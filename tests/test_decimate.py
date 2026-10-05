"""Decimation fidelity + normal-map bake + skin-preserving LODs.

The regressions these pin are the ones that actually shipped:

* `fast_simplification` collapses against an ABSOLUTE error threshold, so a mesh authored in metres
  decimates almost randomly. `test_unit_normalization_*` is the guard — the same mesh at two scales
  must produce the same quality, which it did NOT before `_SIMPLIFY_SPAN`.
* The head used to keep ~12% of the vertex budget. `test_importance_warp_*` pins the allocation and,
  just as importantly, that the warp stays a smooth map (a per-vertex scale tore the mesh and the
  target count was never reached).
* A normal map is only meaningful in the tangent frame it was baked against, so the bake MUST ship
  its own NORMAL/TANGENT — `test_bake_writes_its_own_basis`.
"""
import glob
import io
import os
import tempfile

import numpy as np
import pytest
from PIL import Image
from pygltflib import GLTF2, Skin, FLOAT, UNSIGNED_BYTE

from retopotool import bake_normals as B
from retopotool import decimate as D

from _fixtures import _bumpy_sphere, _write_glb, _write_skinned_glb


@pytest.fixture(scope="module")
def sphere():
    return _bumpy_sphere()


@pytest.fixture(scope="module")
def tmpdir_mod():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _read(path):
    g = GLTF2().load(path)
    blob = g.binary_blob()
    prim = g.meshes[0].primitives[0]
    out = {"gltf": g, "blob": blob, "prim": prim,
           "P": D._acc(g, blob, prim.attributes.POSITION),
           "F": D._acc(g, blob, prim.indices).reshape(-1, 3)}
    for name in ("NORMAL", "TANGENT", "TEXCOORD_0"):
        i = getattr(prim.attributes, name, None)
        out[name] = D._acc(g, blob, i) if i is not None else None
    return out


def _p2s(Q, V, F):
    import igl
    d2, _, _ = igl.point_mesh_squared_distance(np.asarray(Q, np.float64),
                                               np.asarray(V, np.float64),
                                               np.asarray(F, np.int64))
    return np.sqrt(d2)


# ------------------------------- unit normalization -----------------------------------------

@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_unit_normalization_makes_quality_scale_invariant(sphere, tmpdir_mod, scale):
    """THE regression guard. Before `_SIMPLIFY_SPAN` the simplifier's absolute error schedule made
    a metre-scale mesh decimate ~3x worse than the identical centimetre-scale one."""
    V, F, UV, N = sphere
    src = _write_glb(os.path.join(tmpdir_mod, f"src_{scale}.glb"), V, F, UV, N, scale=scale)
    dst = os.path.join(tmpdir_mod, f"out_{scale}.glb")
    st = D.decimate_source_glb(src, dst, 0.25)

    a = _read(src)
    b = _read(dst)
    # error normalized by the model's own size, so the two scales are directly comparable
    span = float(np.linalg.norm(a["P"].max(0) - a["P"].min(0)))
    rel = _p2s(a["P"], b["P"], b["F"]) / span
    assert st["triangles_after"] <= st["triangles_before"] * 0.30
    assert np.percentile(rel, 95) < 2.5e-3, f"scale {scale}: p95 relative error {np.percentile(rel,95)}"


def test_unit_normalization_two_scales_agree(sphere, tmpdir_mod):
    V, F, UV, N = sphere
    errs = {}
    for scale in (1.0, 100.0):
        src = _write_glb(os.path.join(tmpdir_mod, f"s2_{scale}.glb"), V, F, UV, N, scale=scale)
        dst = os.path.join(tmpdir_mod, f"o2_{scale}.glb")
        D.decimate_source_glb(src, dst, 0.25)
        a, b = _read(src), _read(dst)
        span = float(np.linalg.norm(a["P"].max(0) - a["P"].min(0)))
        errs[scale] = float(np.percentile(_p2s(a["P"], b["P"], b["F"]) / span, 95))
    lo, hi = sorted(errs.values())
    assert hi <= lo * 1.35, f"quality depends on authoring units: {errs}"


# ------------------------------- importance warp --------------------------------------------

def _jacobian_dets(fn, box_lo, box_hi, n=18, h=1e-5):
    """Determinant of a warp's Jacobian on a VOLUMETRIC grid.

    Two traps this avoids, both of which produced confidently wrong answers while developing:
      * a surface vertex's 1-ring is coplanar, so a Jacobian fitted to it is rank-deficient and its
        determinant is noise;
      * a triangle's NORMAL is transformed by the inverse-transpose, so it rotates freely under
        shear — a reversed face normal is not evidence of a fold.
    The determinant on volumetric samples is the actual test.
    """
    g = np.stack(np.meshgrid(*[np.linspace(box_lo[i], box_hi[i], n) for i in range(3)],
                             indexing="ij"), axis=-1).reshape(-1, 3)
    J = np.zeros((len(g), 3, 3))
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        J[:, :, k] = (fn(g + d) - fn(g - d)) / (2 * h)
    return np.linalg.det(J)


def test_axis_density_warp_cannot_fold():
    """The warp's core guarantee: det(J) = s**3 > 0 everywhere, so no region can turn inside out.

    This is why the density formulation replaced the obvious one (scale a region about its centroid
    with a smooth falloff), which has no such bound — the rejected form is included here so the
    contrast is a fact in the suite rather than a claim in a comment.
    """
    lo = np.array([-0.83, 0.0, -0.35])
    hi = np.array([0.83, 1.90, 0.35])
    grid = np.linspace(lo[1], hi[1], 512)
    y_head = lo[1] + 0.84 * (hi[1] - lo[1])
    band = 0.08 * (hi[1] - lo[1])
    dens = 1.0 + 2.0 * D._smoothstep((grid - (y_head - band)) / band)
    centre = np.array([0.0, 1.75, 0.0])

    # Probe strictly inside the density grid: np.interp clamps outside it, which halves the
    # one-sided finite difference at the boundary and is an artifact of the probe, not the map.
    pad = 0.02 * (hi - lo)
    det = _jacobian_dets(lambda X: D._axis_density_warp(X, 1, dens, grid, centre=centre),
                         lo + pad, hi - pad)
    assert (det > 0).all(), f"density warp folded: {(det <= 0).sum()} negative determinants"
    # det = s**3, and the density peaks at 1 + 2 = 3.
    assert det.max() <= 3.0 ** 3 * 1.02, f"magnification {det.max()} exceeds the density cubed"

    def centroid_scale(X):
        w = D._smoothstep((X[:, 1] - (y_head - band)) / band)
        return X + (X - centre) * (2.0 * w)[:, None]

    bad = _jacobian_dets(centroid_scale, lo + pad, hi - pad)
    assert (bad <= 0).any(), (
        "the rejected centroid formulation no longer folds — if that is genuinely true, this "
        "contrast test is obsolete; it is here because it DID fold (13.4% of the character's area)")


def test_importance_warp_cannot_fold(sphere):
    V, _, _, _ = sphere
    V = V.astype(np.float64)
    Q, head, _steps = D._importance_warp(V)
    assert head.sum() > 0
    lo, hi = V.min(0), V.max(0)
    # Re-derive the same density fields the warp built, then differentiate THOSE. Calling
    # _importance_warp on perturbed points would re-derive the bands from the perturbed bbox and
    # measure the identity map instead.
    band = max(D._WARP_BAND_FRAC * (hi[1] - lo[1]), 1e-9)
    y_head = lo[1] + D._HEAD_Y_FRAC * (hi[1] - lo[1])
    gy = np.linspace(lo[1], hi[1], 512)
    s = 1.0 + (D._HEAD_BOOST - 1.0) * D._smoothstep((gy - (y_head - band)) / band)
    pad = 0.02 * (hi - lo)
    det = _jacobian_dets(
        lambda X: D._axis_density_warp(X, 1, s, gy, centre=V[head].mean(0)), lo + pad, hi - pad)
    assert (det > 0).all(), f"head warp folded: {(det <= 0).sum()} negative determinants"


def test_importance_warp_gives_the_head_more_of_the_budget(sphere, tmpdir_mod):
    V, F, UV, N = sphere
    src = _write_glb(os.path.join(tmpdir_mod, "warp_src.glb"), V, F, UV, N)
    dst_on = os.path.join(tmpdir_mod, "warp_on.glb")
    dst_off = os.path.join(tmpdir_mod, "warp_off.glb")
    st_on = D.decimate_source_glb(src, dst_on, 0.25)
    old_head, old_hand = D._HEAD_BOOST, D._HAND_BOOST
    D._HEAD_BOOST, D._HAND_BOOST = 1.0, 1.0          # uniform allocation
    try:
        st_off = D.decimate_source_glb(src, dst_off, 0.25)
    finally:
        D._HEAD_BOOST, D._HAND_BOOST = old_head, old_hand

    # The gain is modest on this fixture and large on a real character (measured 13.8% -> 22.7%,
    # i.e. 1.64x, on the reported 3.0M-tri upload): the test sphere concentrates its fine relief
    # across the whole upper hemisphere, so uniform QEM already spends most of the budget there.
    assert st_on["head_vertex_share"] > st_off["head_vertex_share"] * 1.15, (
        f"warp did not reallocate: {st_on['head_vertex_share']} vs {st_off['head_vertex_share']}")
    # and it must not have paid for it by missing the triangle target
    assert st_on["triangles_after"] <= st_off["triangles_after"] * 1.05

    # Head accuracy must not be PAID FOR — the extra vertices have to earn their keep. Only a
    # tie is asserted here because this fixture no longer separates the two: since the decimator
    # started emitting the collapse optimum both variants reproduce this smooth sphere to within
    # 7% of each other (0.00056 vs 0.00052). A real character still shows the gain the warp exists
    # for — measured at 30k on a 399k-tri character, head p95 0.72 mm with the warp vs 0.82 mm
    # without, and the head's share of surviving vertices 0.50 vs 0.31.
    a = _read(src)
    lo, hi = a["P"].min(0), a["P"].max(0)
    head = a["P"][:, 1] > lo[1] + 0.84 * (hi[1] - lo[1])
    on, off = _read(dst_on), _read(dst_off)
    e_on = np.percentile(_p2s(a["P"][head], on["P"], on["F"]), 95)
    e_off = np.percentile(_p2s(a["P"][head], off["P"], off["F"]), 95)
    assert e_on < e_off * 1.15, f"head error regressed: {e_on} vs {e_off}"


def test_inverse_importance_warp_round_trips(sphere):
    """The emitted position is computed in warp space and mapped back through this inverse, so any
    error here lands directly in the character's geometry."""
    V, _, _, _ = sphere
    V = V.astype(np.float64)
    Q, head, steps = D._importance_warp(V)
    assert steps, "the fixture should exercise at least the head warp"
    assert not np.allclose(Q, V), "warp was a no-op; the round trip would prove nothing"
    back = D._inverse_importance_warp(Q, steps)
    span = float(np.linalg.norm(V.max(0) - V.min(0)))
    assert np.abs(back - V).max() < 1e-9 * span, (
        f"inverse warp is not exact: max error {np.abs(back - V).max():.3e}")


def test_importance_warp_degrades_to_identity_without_a_head(sphere):
    """A flat/degenerate input must not get a guessed head band."""
    V = np.zeros((100, 3), dtype=np.float64)
    V[:, 0] = np.linspace(0, 1, 100)
    Q, head, _steps = D._importance_warp(V)
    assert not head.any()
    assert np.allclose(Q, V)


# ------------------------------- the bake ---------------------------------------------------

def test_bake_writes_its_own_basis_and_a_normal_map(sphere, tmpdir_mod):
    V, F, UV, N = sphere
    src = _write_glb(os.path.join(tmpdir_mod, "bake_src.glb"), V, F, UV, N)
    dst = os.path.join(tmpdir_mod, "bake_out.glb")
    st = D.decimate_source_glb(src, dst, 0.2)
    assert st["normal_map_baked"] is True
    assert st["bake_resolution"] >= B.BAKE_MIN_RES

    b = _read(dst)
    assert b["NORMAL"] is not None and b["TANGENT"] is not None
    n, t = b["NORMAL"].astype(np.float64), b["TANGENT"].astype(np.float64)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-3), "non-unit NORMAL"
    assert np.allclose(np.linalg.norm(t[:, :3], axis=1), 1.0, atol=1e-3), "non-unit TANGENT"
    assert set(np.unique(t[:, 3])).issubset({-1.0, 1.0}), "tangent handedness must be +/-1"
    assert np.abs((n * t[:, :3]).sum(1)).max() < 1e-3, "TANGENT not orthogonal to NORMAL"

    g = b["gltf"]
    mat = g.materials[0]
    assert mat.normalTexture is not None
    ti = B._texture_image_index_compat(g, mat.normalTexture.index)
    img = g.images[ti]
    assert img.mimeType == "image/png", "a normal map must never be JPEG"
    bv = g.bufferViews[img.bufferView]
    raw = bytes(b["blob"][bv.byteOffset:bv.byteOffset + bv.byteLength])
    arr = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32) / 255.0
    assert arr.shape[0] == arr.shape[1] == st["bake_resolution"]
    v = arr * 2.0 - 1.0
    assert v[..., 2].min() >= B.MIN_Z - 0.02, "encoded a normal below the tangent plane (renders black)"
    # the map must carry real relief, not be a flat 128/128/255 field
    assert np.linalg.norm(v[..., :2], axis=-1).mean() > 0.01, "baked map is flat — nothing was baked"


def test_bake_recovers_the_high_res_normals(sphere, tmpdir_mod):
    """The point of the whole pass: the low-poly + its baked map must shade like the high-poly."""
    V, F, UV, N = sphere
    src = _write_glb(os.path.join(tmpdir_mod, "rec_src.glb"), V, F, UV, N)
    baked = os.path.join(tmpdir_mod, "rec_baked.glb")
    plain = os.path.join(tmpdir_mod, "rec_plain.glb")
    D.decimate_source_glb(src, baked, 0.2)
    os.environ["RETOPO_OPT_BAKE_NORMALS"] = "0"
    try:
        D.decimate_source_glb(src, plain, 0.2)
    finally:
        os.environ.pop("RETOPO_OPT_BAKE_NORMALS", None)

    import igl
    a = _read(src)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(a["P"]), 4000, replace=False)
    Q = a["P"][idx].astype(np.float64)
    ref = a["NORMAL"][idx].astype(np.float64)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)

    def shaded(path):
        m = _read(path)
        P = m["P"].astype(np.float64)
        Fi = m["F"].astype(np.int64)
        _, fid, pts = igl.point_mesh_squared_distance(Q, P, Fi)
        fid = np.asarray(fid, np.int64)
        tri = P[Fi[fid]]
        v0, v1, v2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], np.asarray(pts) - tri[:, 0]
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-20, 1e-20, d00 * d11 - d01 * d01)
        bb = (d11 * d20 - d01 * d21) / den
        cc = (d00 * d21 - d01 * d20) / den
        w = np.stack([1 - bb - cc, bb, cc], 1)
        cor = Fi[fid]
        n = (m["NORMAL"].astype(np.float64)[cor] * w[:, :, None]).sum(1)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-20
        g = m["gltf"]
        mat = g.materials[0]
        if mat.normalTexture is not None and m["TANGENT"] is not None:
            ti = B._texture_image_index_compat(g, mat.normalTexture.index)
            im = g.images[ti]
            bv = g.bufferViews[im.bufferView]
            nmap = B.decode_image(bytes(m["blob"][bv.byteOffset:bv.byteOffset + bv.byteLength]))
            T = m["TANGENT"].astype(np.float64)
            t = (T[cor, :3] * w[:, :, None]).sum(1)
            t -= n * (n * t).sum(1, keepdims=True)
            t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-20
            bt = np.cross(n, t) * np.sign((T[cor, 3] * w).sum(1))[:, None]
            ts = B._sample_bilinear(nmap, (m["TEXCOORD_0"].astype(np.float64)[cor]
                                           * w[:, :, None]).sum(1)) * 2 - 1
            n = t * ts[:, 0:1] + bt * ts[:, 1:2] + n * ts[:, 2:3]
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-20
        return n

    def deg(n):
        return np.degrees(np.arccos(np.clip((n * ref).sum(1), -1, 1)))

    e_baked = np.median(deg(shaded(baked)))
    e_plain = np.median(deg(shaded(plain)))
    assert e_baked < e_plain, f"bake made shading worse: {e_baked:.2f} vs {e_plain:.2f}"
    assert e_baked < 5.0, f"baked shading error too high: {e_baked:.2f} deg"


def test_bake_composes_with_the_source_normal_map(sphere, tmpdir_mod):
    """Meshy ships a normal map of micro-detail no geometric bake can reproduce. Overwriting it
    would throw that away, so the bake must COMPOSE."""
    V, F, UV, N = sphere
    res = 512
    yy, xx = np.mgrid[0:res, 0:res]
    tsx = 0.6 * np.sin(xx / 4.0)                      # a strong, unmistakable ripple
    tsy = np.zeros_like(tsx)
    tsz = np.sqrt(np.clip(1 - tsx ** 2, 0, 1))
    src_map = np.clip(np.rint((np.stack([tsx, tsy, tsz], -1) * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
    src = _write_glb(os.path.join(tmpdir_mod, "comp_src.glb"), V, F, UV, N, normal_map=src_map)
    dst = os.path.join(tmpdir_mod, "comp_out.glb")
    st = D.decimate_source_glb(src, dst, 0.2)
    assert st["bake"][0]["composed_with_source_map"] is True

    b = _read(dst)
    g = b["gltf"]
    ti = B._texture_image_index_compat(g, g.materials[0].normalTexture.index)
    bv = g.bufferViews[g.images[ti].bufferView]
    out = B.decode_image(bytes(b["blob"][bv.byteOffset:bv.byteOffset + bv.byteLength]))
    v = out * 2 - 1
    # the source's ripple amplitude must survive into the output, not be flattened away
    assert np.percentile(np.linalg.norm(v[..., :2], axis=-1), 90) > 0.15, (
        "source normal-map detail was discarded instead of composed")


def test_bake_can_be_disabled_and_still_produces_a_valid_mesh(sphere, tmpdir_mod):
    V, F, UV, N = sphere
    src = _write_glb(os.path.join(tmpdir_mod, "off_src.glb"), V, F, UV, N)
    dst = os.path.join(tmpdir_mod, "off_out.glb")
    os.environ["RETOPO_OPT_BAKE_NORMALS"] = "0"
    try:
        st = D.decimate_source_glb(src, dst, 0.25)
    finally:
        os.environ.pop("RETOPO_OPT_BAKE_NORMALS", None)
    assert st["normal_map_baked"] is False
    b = _read(dst)
    assert len(b["P"]) > 0 and b["F"].max() < len(b["P"])


# ------------------------------- invariants that must not regress ----------------------------

def _uv_at(Q, V, F, UV):
    """The source parameterization sampled at each point of `Q`: barycentric UV of its closest
    point on the source surface. This is what a decimated vertex's UV SHOULD be."""
    import igl
    _, fi, cp = igl.point_mesh_squared_distance(np.asarray(Q, np.float64),
                                                np.asarray(V, np.float64),
                                                np.asarray(F, np.int64))
    T = np.asarray(V, np.float64)[np.asarray(F, np.int64)[fi]]
    e1 = T[:, 1] - T[:, 0]
    e2 = T[:, 2] - T[:, 0]
    q = cp - T[:, 0]
    d00 = (e1 * e1).sum(1); d01 = (e1 * e2).sum(1); d11 = (e2 * e2).sum(1)
    d20 = (q * e1).sum(1); d21 = (q * e2).sum(1)
    den = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-30, 1e-30, d00 * d11 - d01 * d01)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1.0 - v - w
    UVa = np.asarray(UV, np.float64)[np.asarray(F, np.int64)[fi]]
    return u[:, None] * UVa[:, 0] + v[:, None] * UVa[:, 1] + w[:, None] * UVa[:, 2]


def test_output_uvs_match_the_surface_at_the_position_emitted(sphere, tmpdir_mod):
    """THE texture-fidelity guard: a wedge's UV must describe where the wedge actually IS.

    This replaced an assertion that every output UV was an input UV, exactly. That invariant was
    true and was the defect: a collapse emits the cluster's quadric-optimal position, and handing
    that position a neighbour's verbatim UV slides the texture by the whole collapse distance.
    Exact copying is still how a wedge picks its atlas CHART (see `_resample_uv_on_source`); the
    value is then re-read from the source surface, which is what this measures.

    It bites hardest where the quadric is cheapest — a flat plate carrying a decal has near-zero
    geometric error, gets stripped hardest, and its lettering turns to mush while every geometric
    metric stays clean. Measured on the character that prompted this (1,062,424 -> 399,470 tris,
    2048px atlas): vertices more than one texel out of place 11.3% -> 1.0%, p99 drift 6.6 -> 0.7
    texels, at an unchanged triangle count.

    The oracle takes the globally closest source triangle, which is only unambiguous because
    `_bumpy_sphere` leaves its phi seam UNSTITCHED and so has no coincident faces to choose between
    (the same property that made it unable to express the seam-skin bug, see
    `character-export-optimize.md`). Do not reuse `_uv_at` on a fixture with a real seam without
    first resolving the chart — `seamed_rig` and `test_lod_uvs_still_split_at_the_seam` cover that
    case instead.
    """
    V, F, UV, N = sphere
    src = _write_glb(os.path.join(tmpdir_mod, "uv_src.glb"), V, F, UV, N)
    dst = os.path.join(tmpdir_mod, "uv_out.glb")
    D.decimate_source_glb(src, dst, 0.25)
    b = _read(dst)
    drift = np.linalg.norm(_uv_at(b["P"], V, F, UV) - b["TEXCOORD_0"], axis=1)
    # in texels of a 2048px atlas, the resolution this pipeline bakes at
    assert np.percentile(drift, 95) * 2048 < 1.0, (
        f"UV p95 drift {np.percentile(drift, 95) * 2048:.2f} texels — the texture is sliding "
        "against the surface")


def test_uv_resample_is_what_removes_the_drift(sphere, tmpdir_mod):
    """The lever, pinned: with RETOPO_UV_RESAMPLE=0 the same mesh keeps the verbatim-copy UVs and the
    drift comes back. Guards against the pass being quietly disabled or made a no-op."""
    import importlib
    V, F, UV, N = sphere
    src = _write_glb(os.path.join(tmpdir_mod, "uv_gate_src.glb"), V, F, UV, N)
    prior = os.environ.get("RETOPO_UV_RESAMPLE")     # restore, so a suite run with the gate off is not
    got = {}                                     # silently switched on for every test after this one
    try:
        for flag in ("0", "1"):
            os.environ["RETOPO_UV_RESAMPLE"] = flag
            importlib.reload(D)
            dst = os.path.join(tmpdir_mod, f"uv_gate_{flag}.glb")
            D.decimate_source_glb(src, dst, 0.25)
            b = _read(dst)
            got[flag] = float(np.percentile(
                np.linalg.norm(_uv_at(b["P"], V, F, UV) - b["TEXCOORD_0"], axis=1), 95)) * 2048
    finally:
        if prior is None:
            os.environ.pop("RETOPO_UV_RESAMPLE", None)
        else:
            os.environ["RETOPO_UV_RESAMPLE"] = prior
        importlib.reload(D)
    assert got["1"] < got["0"] * 0.25, f"resample did not reduce UV drift: {got} texels p95"


def test_uv_survives_a_flat_panel_decimated_to_nothing(tmpdir_mod):
    """The complaint in miniature: a broad, gently domed plate with a LINEAR uv map.

    Its quadric error is near zero everywhere, so the simplifier strips it hardest of anything on a
    character — which is exactly what happens to an armour plate carrying a logo. The parameterization
    is known in closed form here, so the assertion is exact rather than statistical: every output
    vertex's UV must equal the affine map evaluated at its own position.
    """
    n = 120
    g = np.linspace(-0.5, 0.5, n)
    X, Z = np.meshgrid(g, g, indexing="ij")
    Y = 0.02 * (1.0 - 4 * X ** 2) * (1.0 - 4 * Z ** 2)      # shallow dome: a well-posed optimum
    V = np.stack([X, Y, Z], -1).reshape(-1, 3).astype(np.float32)
    UV = np.stack([X + 0.5, Z + 0.5], -1).reshape(-1, 2).astype(np.float32)
    idx = np.arange(n * n).reshape(n, n)
    a, b_, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    F = np.concatenate([np.stack([a, b_, c], -1).reshape(-1, 3),
                        np.stack([a, c, d], -1).reshape(-1, 3)]).astype(np.uint32)
    N = np.tile(np.array([0, 1, 0], np.float32), (len(V), 1))
    src = _write_glb(os.path.join(tmpdir_mod, "panel.glb"), V, F, UV, N)
    dst = os.path.join(tmpdir_mod, "panel_out.glb")
    D.decimate_source_glb(src, dst, 0.05)
    b = _read(dst)
    P = np.asarray(b["P"], np.float64)
    expect = np.stack([P[:, 0] + 0.5, P[:, 2] + 0.5], -1)
    err = np.abs(np.asarray(b["TEXCOORD_0"], np.float64) - expect).max()
    # one texel of a 2048px atlas
    assert err < 1.0 / 2048, f"UV is off the affine map by {err * 2048:.2f} texels"


def test_output_positions_lie_on_the_source_surface(sphere, tmpdir_mod):
    """Merged vertices sit at the quadric optimum, so they are NOT source vertices — but they must
    still lie on the source SURFACE. (This replaced a test asserting the output was a vertex subset;
    that pinned the bug in `test_decimation_does_not_invert_faces`.)"""
    V, F, UV, N = sphere
    src = _write_glb(os.path.join(tmpdir_mod, "pos_src.glb"), V, F, UV, N)
    dst = os.path.join(tmpdir_mod, "pos_out.glb")
    D.decimate_source_glb(src, dst, 0.25)
    a, b = _read(src), _read(dst)
    span = float(np.linalg.norm(a["P"].max(0) - a["P"].min(0)))
    d = _p2s(b["P"], a["P"], a["F"])
    assert np.percentile(d, 99) < 0.01 * span, (
        f"output vertices drifted off the source surface: p99 {np.percentile(d, 99):.5f} "
        f"of a {span:.3f} span")


def test_untouched_vertices_keep_their_exact_position(sphere, tmpdir_mod):
    """A vertex that absorbed no neighbour must come out bit-identical: the optimal-position change
    may only move vertices that actually merged."""
    V, F, UV, N = sphere
    src = _write_glb(os.path.join(tmpdir_mod, "keep_src.glb"), V, F, UV, N)
    dst = os.path.join(tmpdir_mod, "keep_out.glb")
    D.decimate_source_glb(src, dst, 0.25)
    a, b = _read(src), _read(dst)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(a["P"].astype(np.float64)).query(b["P"].astype(np.float64))
    exact = d == 0.0
    assert exact.sum() > 0.05 * len(b["P"]), (
        f"only {exact.sum()}/{len(b['P'])} output vertices are bit-identical to a source vertex — "
        "un-collapsed vertices are being moved")


def test_uv_survives_a_high_valence_pole(tmpdir_mod):
    """A pole is the case the capped adjacency table can truncate.

    `_resample_uv_on_source` sizes its ring/incidence tables from the mesh's own maximum degree and
    only clips at `_UV_RESAMPLE_VALENCE_MAX`, so a 48-spoke fan is truncated by design — and a
    truncated ring can drop the very edge the walk needed. This pins that the clip degrades
    gracefully rather than silently: same closed-form UV assertion as the flat panel, on a disc whose
    centre vertex has eight times a regular vertex's valence.
    """
    n_spoke, n_ring = 48, 26
    th = np.linspace(0, 2 * np.pi, n_spoke, endpoint=False)
    rr = np.linspace(0.5 / n_ring, 0.5, n_ring)
    R, T = np.meshgrid(rr, th, indexing="ij")
    X = (R * np.cos(T)).ravel()
    Z = (R * np.sin(T)).ravel()
    X = np.concatenate([[0.0], X])
    Z = np.concatenate([[0.0], Z])
    Y = 0.05 * (0.25 - X ** 2 - Z ** 2)                  # shallow dome: a well-posed optimum
    V = np.stack([X, Y, Z], -1).astype(np.float32)
    UV = np.stack([X + 0.5, Z + 0.5], -1).astype(np.float32)
    idx = (1 + np.arange(n_ring * n_spoke).reshape(n_ring, n_spoke))
    fan = np.stack([np.zeros(n_spoke, int), idx[0], np.roll(idx[0], -1)], -1)
    a = idx[:-1]
    b_ = np.roll(idx[:-1], -1, axis=1)
    c = np.roll(idx[1:], -1, axis=1)
    d = idx[1:]
    F = np.concatenate([fan,
                        np.stack([a, b_, c], -1).reshape(-1, 3),
                        np.stack([a, c, d], -1).reshape(-1, 3)]).astype(np.uint32)
    N = np.tile(np.array([0, 1, 0], np.float32), (len(V), 1))
    src = _write_glb(os.path.join(tmpdir_mod, "pole.glb"), V, F, UV, N)
    dst = os.path.join(tmpdir_mod, "pole_out.glb")
    D.decimate_source_glb(src, dst, 0.15)
    b = _read(dst)
    P = np.asarray(b["P"], np.float64)
    expect = np.stack([P[:, 0] + 0.5, P[:, 2] + 0.5], -1)
    err = np.abs(np.asarray(b["TEXCOORD_0"], np.float64) - expect).max()
    assert err < 1.0 / 2048, f"UV is off the affine map by {err * 2048:.2f} texels at a pole"


def test_decimation_does_not_invert_faces(sphere):
    """THE HOLES BUG. Emitting the surviving original vertex instead of the collapse cluster's
    quadric optimum invalidates the simplifier's own normal-flip test (it ran at the optimum), and
    faces turn inside out. Materials are single-sided, so an inverted face is an invisible one —
    that was the reported "holes and floating shards" in the LODs. Measured on a real 399k-tri
    character: 1.75% / 3.28% / 4.07% of faces inverted at 100k / 30k / 15k before the fix.

    Each output face descends from a known ORIGINAL face (`stats['ancestor_faces']`), so this
    compares against the normal that face should still have — a nearest-face lookup would
    mismatch exactly where the mesh is worst.
    """
    V, F, UV, N = sphere
    P = V.astype(np.float32)
    Fi = F.astype(np.int64)

    def face_normals(pos, tri):
        n = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
        a = np.linalg.norm(n, axis=1)
        return n / np.maximum(a, 1e-30)[:, None], a

    src_n, _ = face_normals(P.astype(np.float64), Fi)
    for ratio in (0.25, 0.05):
        stats = {}
        pos, faces, _ = D._decimate_primitive(P, Fi, {"TEXCOORD_0": UV}, ratio, stats=stats)
        out_n, _area = face_normals(pos.astype(np.float64), faces)
        inverted = (out_n * src_n[stats["ancestor_faces"]]).sum(1) < 0
        assert not inverted.any(), (
            f"at ratio {ratio}: {inverted.sum()}/{len(faces)} faces are backface-culled holes")
        # The solidity pass must not have bought that by tearing the mesh open instead.
        assert stats["winding_flipped"] < 0.001 * len(faces), (
            f"at ratio {ratio}: {stats['winding_flipped']} faces needed a winding flip — the "
            "position repair is not doing its job")


def test_flap_removal_refuses_to_tear_the_surface():
    """`_drop_zero_volume_flaps` is the only repair that DELETES faces, so it is the only one that
    can open a real hole. It may fire only where each of the pair's edges is used by exactly the
    pair (the edge disappears with them) or by two more faces (it stays manifold). At multiplicity
    three, removal would strand one face on that edge — a boundary, i.e. the very hole we are
    trying to remove."""
    # A closed tetrahedron, plus an isolated flap on three unused vertices.
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [5, 0, 0], [6, 0, 0], [5, 1, 0]], dtype=np.float64)
    tetra = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]

    # (a) isolated opposite-wound pair: every edge is used only by the pair -> removed
    faces = np.array(tetra + [[4, 5, 6], [4, 6, 5]], dtype=np.int64)
    keep = D._drop_zero_volume_flaps(pos, faces)
    assert keep.tolist() == [True] * 4 + [False, False]

    # (b) the flap sits ON the closed surface: its edges would be left with one face -> kept
    faces = np.array(tetra + [[0, 2, 1]], dtype=np.int64)
    keep = D._drop_zero_volume_flaps(pos, faces)
    assert keep.all(), "removing this pair would have torn a hole in the tetrahedron"

    # (c) same winding is a true duplicate, not a zero-volume flap -> left alone
    faces = np.array(tetra + [[4, 5, 6], [4, 5, 6]], dtype=np.int64)
    keep = D._drop_zero_volume_flaps(pos, faces)
    assert keep.all()


def test_no_coincident_faces_survive_decimation(sphere, tmpdir_mod):
    """Collapses fold the surface back on itself; every such pair measured on a real character was
    opposite-wound, and they took the surrounding non-manifold edges with them (30k LOD: 48 pairs
    and 81 non-manifold edges -> 0 and 14)."""
    V, F, UV, N = sphere
    src = _write_glb(os.path.join(tmpdir_mod, "solid_src.glb"), V, F, UV, N)
    dst = os.path.join(tmpdir_mod, "solid_out.glb")
    D.decimate_source_glb(src, dst, 0.1)
    b = _read(dst)
    P, Fo = b["P"].astype(np.float64), b["F"].astype(np.int64)
    span = float(np.linalg.norm(P.max(0) - P.min(0)))
    _, w = np.unique(np.round(P / (span * 1e-6)).astype(np.int64), axis=0, return_inverse=True)
    w = w.reshape(-1)
    nw = int(w.max()) + 1
    tri = np.sort(w[Fo], 1)
    _, fc = np.unique((tri[:, 0] * nw + tri[:, 1]) * nw + tri[:, 2], return_counts=True)
    assert int((fc > 1).sum()) == 0, f"{(fc > 1).sum()} coincident faces survived"


def test_rejects_skinned_input(sphere, tmpdir_mod):
    V, F, UV, N = sphere
    src = _write_glb(os.path.join(tmpdir_mod, "skin_src.glb"), V, F, UV, N)
    g = GLTF2().load(src)
    from pygltflib import Skin
    g.skins = [Skin(joints=[0])]
    g.save(src)
    with pytest.raises(ValueError, match="rigged/animated"):
        D.decimate_source_glb(src, os.path.join(tmpdir_mod, "skin_out.glb"), 0.5)


def test_dilation_fills_past_island_borders():
    rgb = np.zeros((16, 16, 3), dtype=np.float32)
    mask = np.zeros((16, 16), dtype=bool)
    rgb[8, 8] = [0.25, 0.75, 1.0]
    mask[8, 8] = True
    out = B._dilate(rgb, mask, iters=4)
    assert out[8, 10, 1] > 0.5, "dilation did not push the value outward"
    assert np.allclose(out[8, 8], [0.25, 0.75, 1.0]), "dilation overwrote a covered texel"


def test_pick_resolution_clamps_to_power_of_two():
    assert B.pick_resolution([]) == B.BAKE_MIN_RES
    assert B.pick_resolution([1024]) == B.BAKE_MIN_RES
    assert B.pick_resolution([4096]) == 4096
    assert B.pick_resolution([16384]) == B.BAKE_MAX_RES


# ==============================================================================================
# Rigged LOD ladder
#
# Every assertion below guards a failure that is SILENT: a stale inverse-bind index, a joint index
# rescaled to 0..1, or a dropped skin accessor all produce a GLB that loads fine and then deforms
# into garbage. None of them raise on their own, which is why they are pinned here.
# ==============================================================================================

@pytest.fixture(scope="module")
def rigged(sphere, tmpdir_mod):
    V, F, UV, N = sphere
    return _write_skinned_glb(os.path.join(tmpdir_mod, "rig.glb"), V, F, UV, N, orphans=3)


def test_unrigged_path_still_rejects_a_skin(rigged, tmpdir_mod):
    """The pre-rig contract must not be loosened by the LOD work."""
    with pytest.raises(ValueError, match="rigged/animated"):
        D.decimate_source_glb(rigged, os.path.join(tmpdir_mod, "nope.glb"), 0.5)


def test_lod_preserves_the_skin(rigged, tmpdir_mod):
    dst = os.path.join(tmpdir_mod, "lod.glb")
    src = _read(rigged)
    st = D.decimate_rigged_glb(rigged, dst, 8000, texture_size=512)
    assert st["triangles_after"] <= 8000 * 1.02

    g = GLTF2().load(dst)
    blob = g.binary_blob()
    prim = g.meshes[0].primitives[0]
    nv = g.accessors[prim.attributes.POSITION].count

    # --- JOINTS_0 must stay INTEGER bone indices ---
    ja = g.accessors[prim.attributes.JOINTS_0]
    assert ja.componentType in (UNSIGNED_BYTE, 5123), (
        "JOINTS_0 was written as a float — the bone indices went through the "
        "'integer attribute -> 0..1' rescale and the rig is destroyed")
    assert not ja.normalized, "JOINTS_0 must not be flagged normalized; indices are not a ratio"
    J = D._acc(g, blob, prim.attributes.JOINTS_0)
    assert np.issubdtype(J.dtype, np.integer)
    njoints = len(g.skins[0].joints)
    assert int(J.max()) < njoints
    # the top bone must still be reachable: a /255 rescale would have collapsed every index to ~0
    src_j = D._acc(GLTF2().load(rigged), _read(rigged)["blob"], _read(rigged)["prim"].attributes.JOINTS_0)
    assert int(J.max()) == int(src_j.max()), "the highest bone index did not survive"

    # --- weights ---
    W = D._acc(g, blob, prim.attributes.WEIGHTS_0).astype(np.float64)
    assert len(W) == nv and len(J) == nv, "skin attribute count != vertex count"
    assert np.allclose(W.sum(1), 1.0, atol=1e-4), "weights no longer sum to 1"
    assert (W.sum(1) > 1e-6).all(), "a vertex ended up with no skin at all"

    # --- the skeleton is untouched, and the IBM accessor was RE-POINTED ---
    assert g.skins[0].joints == src["gltf"].skins[0].joints
    ibm = g.accessors[g.skins[0].inverseBindMatrices]
    assert ibm.type == "MAT4" and ibm.componentType == FLOAT, (
        "inverse-bind accessor is cross-wired — the repack rebuilt g.accessors and left the "
        "skin pointing at whatever now sits at the old index")
    assert ibm.count == njoints

    # --- indices in range ---
    idx = D._acc(g, blob, prim.indices)
    assert int(idx.max()) < nv


def test_lod_drops_orphaned_accessors(rigged, tmpdir_mod):
    """A rig writer that appends instead of repacking ships ~25% dead geometry. Copying it
    forward made a 15k-triangle LOD weigh 11 MB instead of 1.7."""
    dst = os.path.join(tmpdir_mod, "lod_orphans.glb")
    D.decimate_rigged_glb(rigged, dst, 8000, texture_size=512, bake=False)
    before = len(GLTF2().load(rigged).accessors)
    after = len(GLTF2().load(dst).accessors)
    assert after < before, f"orphaned accessors were carried into the LOD ({before} -> {after})"
    assert os.path.getsize(dst) < os.path.getsize(rigged)


def test_lod_bakes_a_normal_map_and_keeps_tangents(rigged, tmpdir_mod):
    """The old export optimizer dropped TANGENT on decimated primitives, which silently disables
    every normal map downstream."""
    dst = os.path.join(tmpdir_mod, "lod_baked.glb")
    st = D.decimate_rigged_glb(rigged, dst, 8000, texture_size=512, bake_reference=rigged)
    assert st["normal_map_baked"] is True
    assert st["bake_resolution"] == 512, "texture_size did not reach the bake"
    m = _read(dst)
    assert m["TANGENT"] is not None, "TANGENT was dropped"
    t = m["TANGENT"].astype(np.float64)
    assert np.allclose(np.linalg.norm(t[:, :3], axis=1), 1.0, atol=1e-3)
    assert set(np.unique(t[:, 3])).issubset({-1.0, 1.0})


def test_lod_bake_improves_shading(rigged, tmpdir_mod):
    """The whole justification for the ladder: geometry error grows, shading error does not."""
    import igl
    baked = os.path.join(tmpdir_mod, "lod_sh_baked.glb")
    plain = os.path.join(tmpdir_mod, "lod_sh_plain.glb")
    D.decimate_rigged_glb(rigged, baked, 6000, texture_size=1024, bake_reference=rigged)
    D.decimate_rigged_glb(rigged, plain, 6000, texture_size=1024, bake=False)

    a = _read(rigged)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(a["P"]), 3000, replace=False)
    Q = a["P"][idx].astype(np.float64)
    ref = a["NORMAL"][idx].astype(np.float64)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)

    def err(path):
        m = _read(path)
        P = m["P"].astype(np.float64)
        Fi = m["F"].astype(np.int64)
        _, fid, pts = igl.point_mesh_squared_distance(Q, P, Fi)
        fid = np.asarray(fid, np.int64)
        tri = P[Fi[fid]]
        v0, v1, v2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], np.asarray(pts) - tri[:, 0]
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-20, 1e-20, d00 * d11 - d01 * d01)
        bb = (d11 * d20 - d01 * d21) / den
        cc = (d00 * d21 - d01 * d20) / den
        w = np.stack([1 - bb - cc, bb, cc], 1)
        cor = Fi[fid]
        n = (m["NORMAL"].astype(np.float64)[cor] * w[:, :, None]).sum(1)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-20
        g = m["gltf"]
        mat = g.materials[0]
        if mat.normalTexture is not None and m["TANGENT"] is not None:
            ti = B._texture_image_index_compat(g, mat.normalTexture.index)
            im = g.images[ti]
            bv = g.bufferViews[im.bufferView]
            nmap = B.decode_image(bytes(m["blob"][bv.byteOffset:bv.byteOffset + bv.byteLength]))
            T = m["TANGENT"].astype(np.float64)
            tt = (T[cor, :3] * w[:, :, None]).sum(1)
            tt -= n * (n * tt).sum(1, keepdims=True)
            tt /= np.linalg.norm(tt, axis=1, keepdims=True) + 1e-20
            bt = np.cross(n, tt) * np.sign((T[cor, 3] * w).sum(1))[:, None]
            ts = B._sample_bilinear(nmap, (m["TEXCOORD_0"].astype(np.float64)[cor]
                                           * w[:, :, None]).sum(1)) * 2 - 1
            n = tt * ts[:, 0:1] + bt * ts[:, 1:2] + n * ts[:, 2:3]
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-20
        return float(np.median(np.degrees(np.arccos(np.clip((n * ref).sum(1), -1, 1)))))

    e_baked, e_plain = err(baked), err(plain)
    assert e_baked < e_plain, f"the bake made LOD shading worse: {e_baked:.2f} vs {e_plain:.2f}"


def test_lod_rejects_multiple_skins(rigged, tmpdir_mod):
    g = GLTF2().load(rigged)
    g.skins = list(g.skins) + [Skin(joints=[1], inverseBindMatrices=g.skins[0].inverseBindMatrices)]
    two = os.path.join(tmpdir_mod, "two_skins.glb")
    g.save(two)
    with pytest.raises(ValueError, match="single skin"):
        D.decimate_rigged_glb(two, os.path.join(tmpdir_mod, "x.glb"), 5000)


# --------------------------------------------------------------------------------------------
# The extension trap
# --------------------------------------------------------------------------------------------
def test_non_glb_output_is_rejected(rigged, tmpdir_mod):
    """pygltflib picks GLB vs JSON off the FILE EXTENSION.

    An atomic-publish temp name like `foo.glb.tmp` was written as a JSON glTF plus a `.bin`
    sidecar and loaded back with NO binary blob — the LOD endpoint never produced a GLB and only
    failed later, inside the skin assertion, with `a bytes-like object is required, not NoneType`.
    Fail at the write instead.
    """
    out = os.path.join(tmpdir_mod, "ext_trap_lod.glb.tmp")

    with pytest.raises(ValueError, match=r"\.glb"):
        D.decimate_rigged_glb(rigged, out, 200, bake=False)

    # ...and nothing was written on the way out.
    assert not os.path.exists(out)
    assert not glob.glob(os.path.join(tmpdir_mod, "ext_trap_lod*.bin"))


# --------------------------------------------------------------------------------------------
# UV-SEAM REGRESSION FIXTURE.
#
# `_bumpy_sphere` leaves its phi seam UNSTITCHED, so it has no two vertices at the same position
# and CANNOT express the defect below. That is precisely why the defect shipped: every test, and
# every static mesh check in the pipeline, was blind to it. This fixture closes the seam the way a
# real texture atlas does — a duplicated meridian, same positions, different U — so a decimator
# that resolves skin per UV wedge instead of per position is caught.
# --------------------------------------------------------------------------------------------
def _seamed_sphere(n_theta=120, n_phi=90, height=1.8):
    th = np.linspace(0.001, np.pi - 0.001, n_theta)
    ph = np.linspace(0, 2 * np.pi, n_phi + 1)          # endpoint INCLUDED -> duplicated meridian
    T, P = np.meshgrid(th, ph, indexing="ij")
    r = 1.0 + 0.02 * np.sin(5 * T) * np.sin(5 * P)
    X = r * np.sin(T) * np.cos(P)
    Y = r * np.cos(T)
    Z = r * np.sin(T) * np.sin(P)
    V = np.stack([X, Y, Z], axis=-1).reshape(-1, 3) * (height / 2.0)
    # The duplicated column carries U=1 while its twin at phi=0 carries U=0: identical position,
    # different UV — a real seam wedge.
    UV = np.stack([P / (2 * np.pi), T / np.pi], axis=-1).reshape(-1, 2)
    idx = np.arange(n_theta * (n_phi + 1)).reshape(n_theta, n_phi + 1)
    a = idx[:-1, :-1]; b = idx[:-1, 1:]; c = idx[1:, 1:]; d = idx[1:, :-1]
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                        np.stack([a, c, d], -1).reshape(-1, 3)])
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    fn = np.cross(e1, e2)
    acc = np.zeros_like(V, dtype=np.float64)
    for k in range(3):
        np.add.at(acc, F[:, k].astype(np.int64), fn)
    N = acc / (np.linalg.norm(acc, axis=1, keepdims=True) + 1e-20)
    return V.astype(np.float32), F.astype(np.uint32), UV.astype(np.float32), N.astype(np.float32)


@pytest.fixture(scope="module")
def seamed_rig(tmpdir_mod):
    V, F, UV, N = _seamed_sphere()
    return _write_skinned_glb(os.path.join(tmpdir_mod, "seamed_rig.glb"), V, F, UV, N, n_joints=8)


def _skin_by_position(path):
    """(number of position-weld groups with >1 member, number whose skin rows disagree)."""
    g = GLTF2().load(path)
    blob = g.binary_blob()
    prim = g.meshes[0].primitives[0]
    P = D._acc(g, blob, prim.attributes.POSITION).astype(np.float64)
    J = D._acc(g, blob, prim.attributes.JOINTS_0).astype(np.int64)
    W = D._acc(g, blob, prim.attributes.WEIGHTS_0).astype(np.float64)
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    nb = len(g.skins[0].joints)
    M = np.zeros((len(J), nb))
    rows = np.arange(len(J))
    for k in range(J.shape[1]):
        np.add.at(M, (rows, J[:, k]), W[:, k])
    key = np.round(P * 1e6).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    st = np.searchsorted(inv[order], np.arange(len(cnt)))
    en = np.r_[st[1:], len(order)]
    multi = np.flatnonzero(cnt > 1)
    bad = 0
    for gi in multi:
        mem = order[st[gi]:en[gi]]
        if np.abs(M[mem] - M[mem][0]).sum(1).max() > 1e-6:
            bad += 1
    return len(multi), bad


def test_seam_fixture_actually_has_coincident_vertices(seamed_rig):
    """Guard the guard: if the fixture ever loses its seam, every assertion below goes vacuous."""
    groups, _ = _skin_by_position(seamed_rig)
    assert groups > 100, f"fixture has only {groups} seam weld groups — it cannot express the defect"


def test_lod_skin_is_identical_across_a_uv_seam(seamed_rig, tmpdir_mod):
    """THE regression. Two vertices at the same position are the same point on the body, so they
    must deform together. Resolving skin per UV wedge gave them different bones — measured on a real
    character at 60% of seam groups, opening a 5 mm (p95) / 15 mm (p99) gap once posed, while every
    static check still reported a watertight, correctly wound, volume-preserving mesh."""
    dst = os.path.join(tmpdir_mod, "seam_lod.glb")
    D.decimate_rigged_glb(seamed_rig, dst, 4000, texture_size=256, bake=False)
    groups, bad = _skin_by_position(dst)
    assert groups > 0
    assert bad == 0, (f"{bad}/{groups} seam weld groups have differing skin — the two sides of a UV "
                      f"seam will tear apart under animation")


def test_lod_uvs_still_split_at_the_seam(seamed_rig, tmpdir_mod):
    """The counterpart: sharing SKIN across a seam must not collapse the UV split, or the texture
    tears instead. Position-welded vertices must still carry more than one UV."""
    dst = os.path.join(tmpdir_mod, "seam_lod_uv.glb")
    D.decimate_rigged_glb(seamed_rig, dst, 4000, texture_size=256, bake=False)
    g = GLTF2().load(dst)
    blob = g.binary_blob()
    prim = g.meshes[0].primitives[0]
    P = D._acc(g, blob, prim.attributes.POSITION).astype(np.float64)
    UV = D._acc(g, blob, prim.attributes.TEXCOORD_0).astype(np.float64)
    key = np.round(P * 1e6).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    split = 0
    for gi in np.flatnonzero(cnt > 1):
        mem = np.flatnonzero(inv == gi)
        if np.abs(UV[mem] - UV[mem][0]).max() > 1e-6:
            split += 1
    assert split > 0, "no position-weld group carries two UVs — the seam split was destroyed"


def test_blended_skin_never_invents_a_joint(seamed_rig, tmpdir_mod):
    """Blending must stay inside the set of bones the collapse cluster actually used, which is what
    keeps `_assert_skin_intact`'s joint-range check meaningful."""
    dst = os.path.join(tmpdir_mod, "seam_lod_j.glb")
    D.decimate_rigged_glb(seamed_rig, dst, 4000, texture_size=256, bake=False)
    src = GLTF2().load(seamed_rig)
    sj = D._acc(src, src.binary_blob(), src.meshes[0].primitives[0].attributes.JOINTS_0)
    g = GLTF2().load(dst)
    j = D._acc(g, g.binary_blob(), g.meshes[0].primitives[0].attributes.JOINTS_0)
    assert set(np.unique(j).tolist()) <= set(np.unique(sj).tolist())


def test_blended_skin_rows_are_valid(seamed_rig, tmpdir_mod):
    dst = os.path.join(tmpdir_mod, "seam_lod_w.glb")
    D.decimate_rigged_glb(seamed_rig, dst, 4000, texture_size=256, bake=False)
    g = GLTF2().load(dst)
    blob = g.binary_blob()
    prim = g.meshes[0].primitives[0]
    W = D._acc(g, blob, prim.attributes.WEIGHTS_0).astype(np.float64)
    assert W.shape[1] == 4
    assert (W >= -1e-6).all(), "a negative skin weight"
    assert np.allclose(W.sum(1), 1.0, atol=1e-5), "skin weights do not sum to 1"


def _slivery_grid(n=24, seed=3):
    """A planar grid jittered hard along X only. The fixed-diagonal triangulation then leaves ~9% of
    faces under a 10 degree minimum angle, and many of those quads want the OTHER diagonal — which
    is exactly the configuration the flip pass exists to fix. A regular sphere grid cannot exercise
    it: its triangles are already well shaped and no flip improves anything."""
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    V = np.stack([gx.ravel() * 1.0, gy.ravel() * 1.0, np.zeros(n * n)], 1)
    V[:, 0] += rng.normal(0, 0.42, len(V))
    V[:, 1] += rng.normal(0, 0.05, len(V))
    idx = np.arange(n * n).reshape(n, n)
    a = idx[:-1, :-1]; b = idx[:-1, 1:]; c = idx[1:, 1:]; d = idx[1:, :-1]
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                        np.stack([a, c, d], -1).reshape(-1, 3)]).astype(np.int64)
    return V, F


def test_edge_flips_move_no_vertex():
    """The quality pass is flips ONLY: it retriangulates, it never relocates. If it ever starts
    moving vertices it silently breaks the exact-copy attribute contract — a vertex would keep its
    ancestor's UV at a position that ancestor never occupied — so pin the invariant directly on
    `_flip_round`. (End to end the written positions CAN differ, because changing the triangulation
    changes which faces the solidity repair afterwards decides to re-place; that is the repair
    moving them, not this pass.)"""
    V, F = _slivery_grid()
    pos = V.copy()
    before = pos.copy()
    Fo, Fraw = F.copy(), F.copy()
    n = D._flip_round(pos, Fo, Fraw, None, None, float(np.cos(np.radians(20.0))), 1.0)
    assert n > 0, "no flips applied — the fixture cannot exercise the invariant"
    assert np.array_equal(pos, before), "the flip pass mutated vertex positions"


def test_edge_flips_keep_the_face_count_and_the_vertex_set():
    """A flip replaces two triangles with two triangles over the same four corners."""
    V, F = _slivery_grid()
    pos = V.copy()
    Fo, Fraw = F.copy(), F.copy()
    before_faces = len(Fo)
    before_verts = set(np.unique(Fo).tolist())
    D._flip_round(pos, Fo, Fraw, None, None, float(np.cos(np.radians(20.0))), 1.0)
    assert len(Fo) == before_faces
    assert set(np.unique(Fo).tolist()) == before_verts
    degen = (Fo[:, 0] == Fo[:, 1]) | (Fo[:, 1] == Fo[:, 2]) | (Fo[:, 0] == Fo[:, 2])
    assert not degen.any(), "a flip produced a degenerate triangle"


def test_edge_flips_never_cross_a_hard_crease():
    """A flip across an armor-plate boundary rounds off the hard surface. Two quads meeting at 90
    degrees: the shared edge is the crease and must survive."""
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                    [0, 0, 1], [1, 0, 1]], dtype=np.float64)
    # flat quad (0,1,3,2) in z=0 and an upright quad (0,1,5,4) sharing edge 0-1
    Fo = np.array([[0, 1, 3], [0, 3, 2], [1, 0, 4], [1, 4, 5]], dtype=np.int64)
    Fraw = Fo.copy()
    before = Fo.copy()
    crease_cos = float(np.cos(np.radians(20.0)))
    D._flip_round(pos, Fo, Fraw, None, None, crease_cos, 1.0)
    shared = {(0, 1)}
    got = set()
    for tri in Fo:
        for x, y in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            got.add((min(int(x), int(y)), max(int(x), int(y))))
    assert shared <= got, f"the 90-degree crease edge was flipped away: {before.tolist()} -> {Fo.tolist()}"


def test_edge_flips_actually_improve_triangle_shape():
    """The pass must earn its runtime: measured on the jittered grid it takes slivers 8.7% -> 6.7%
    and min-angle p5 4.73 -> 5.91 degrees, without moving a vertex."""
    V, F = _slivery_grid()
    pos = V.copy()
    Fo, Fraw = F.copy(), F.copy()
    before = D._tri_min_angle(pos, Fo)
    for _ in range(4):
        if D._flip_round(pos, Fo, Fraw, None, None, float(np.cos(np.radians(20.0))), 1.0) == 0:
            break
    after = D._tri_min_angle(pos, Fo)
    assert (after < 10).mean() < (before < 10).mean(), "the flip pass did not reduce slivers"
    assert np.percentile(after, 5) > np.percentile(before, 5)


def test_edge_flips_never_make_an_edge_non_manifold():
    """Two flips in one round can share BOTH opposite corners and propose the identical new
    diagonal. The per-face `used` mask does not stop that (the pairs are face-disjoint) and the
    vectorised duplicate check reads the edge table as it was before the round, so accepting both
    would put four faces on one edge. Found by external review, not by the shape metrics — a
    non-manifold edge does not move any of them."""
    V, F = _slivery_grid()
    pos = V.copy()
    Fo, Fraw = F.copy(), F.copy()
    for _ in range(4):
        if D._flip_round(pos, Fo, Fraw, None, None, float(np.cos(np.radians(20.0))), 1.0) == 0:
            break
    E = np.sort(np.concatenate([Fo[:, [0, 1]], Fo[:, [1, 2]], Fo[:, [2, 0]]]), axis=1)
    _, counts = np.unique(E, axis=0, return_counts=True)
    assert counts.max() <= 2, f"{int((counts > 2).sum())} edges have more than two faces"
    faces = np.sort(Fo, axis=1)
    _, fc = np.unique(faces, axis=0, return_counts=True)
    assert fc.max() == 1, f"{int((fc > 1).sum())} duplicate faces after flipping"


def test_eight_influence_rig_is_refused_not_mangled(rigged, tmpdir_mod):
    """The writer normalizes each WEIGHTS_n row to 1 on its own, so carrying a second influence set
    through would emit weights summing to 2 — and `_assert_skin_intact` only inspects set 0."""
    g = GLTF2().load(rigged)
    prim = g.meshes[0].primitives[0]
    prim.attributes.JOINTS_1 = prim.attributes.JOINTS_0
    prim.attributes.WEIGHTS_1 = prim.attributes.WEIGHTS_0
    src8 = os.path.join(tmpdir_mod, "rig8.glb")
    g.save(src8)
    with pytest.raises(ValueError, match="8 skin influences"):
        D.decimate_rigged_glb(src8, os.path.join(tmpdir_mod, "lod8.glb"), 4000, bake=False)
