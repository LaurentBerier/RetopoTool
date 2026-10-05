"""Static meshes, props and environment pieces — the inputs this tool mostly sees.

Each test pins a failure the character-only pipeline had on real prop layouts: cracks between the
materials of one mesh, hard edges smoothed away, accessors shared between primitives, tiling UVs
baked into garbage, a floor that defeated the bake cage, two meshes overwriting one material's map,
and custom attributes left pointing into the old accessor table.
"""
import io
import os

import numpy as np
import pytest
from PIL import Image
from pygltflib import GLTF2, UNSIGNED_SHORT

from _prop_fixtures import (crate_sides, flat_normal_map, floor_tile, merge, read_prims,
                            write_scene)
from retopotool import (NothingToDecimate, build_lod_ladder, decimate_source_glb, mesh_stats,
                        optimize_glb)
from retopotool.gltf_io import _acc, load_triangles
from retopotool.mesh_quality import topology_report, weld_by_position


@pytest.fixture(scope="module")
def sides():
    return crate_sides(n=40)


@pytest.fixture(scope="module")
def tmpd(tmp_path_factory):
    return str(tmp_path_factory.mktemp("props"))


def _prim(part, material=0):
    V, F, UV, N = part
    return {"V": V, "F": F, "UV": UV, "N": N, "material": material}


@pytest.fixture(scope="module")
def two_material_crate(sides, tmpd):
    a, b = merge(sides[:3]), merge(sides[3:])
    return write_scene(os.path.join(tmpd, "crate2.glb"), [[_prim(a, 0), _prim(b, 1)]],
                       materials=[{}, {}])


def _boundary_edges(path):
    """Boundary edges of every mesh, each mesh welded across its own primitives."""
    _, prims = read_prims(path)
    total = 0
    for mi in {p[0] for p in prims}:
        group = [p for p in prims if p[0] == mi]
        P = np.concatenate([p[2] for p in group])
        off, F = 0, []
        for p in group:
            F.append(p[3] + off)
            off += len(p[2])
        Pw, Fw, _ = weld_by_position(P, np.concatenate(F))
        total += topology_report(Pw, Fw, welded=True)["boundary_edges"]
    return total


def _box_edge_normals(path):
    """Normals carried by vertices on the crate's 12 edges: a hard edge keeps them axis-aligned."""
    _, prims = read_prims(path)
    P = np.concatenate([p[2] for p in prims])
    N = np.concatenate([p[4]["NORMAL"] for p in prims]).astype(np.float64)
    s = np.sort(np.abs(P), axis=1)
    edge = s[:, 1] > 0.495                       # two coordinates on the box surface
    return np.abs(N[edge]).max(1)


# ---- one surface, several materials --------------------------------------------------------

def test_material_borders_do_not_crack(two_material_crate, tmpd):
    assert _boundary_edges(two_material_crate) == 0
    dst = os.path.join(tmpd, "crate2_opt.glb")
    st = optimize_glb(two_material_crate, dst, ratio=0.1)
    assert st["quality"]["boundary_edges"] == 0
    assert _boundary_edges(dst) == 0, "the two materials' border opened up"
    _, prims = read_prims(dst)
    assert [p[5] for p in prims] == [0, 1], "every primitive keeps its own material"
    assert st["triangles_after"] < 0.15 * st["triangles_before"]


def test_hard_edges_survive_the_bake(two_material_crate, tmpd):
    dst = os.path.join(tmpd, "crate2_hard.glb")
    st = optimize_glb(two_material_crate, dst, ratio=0.1)
    assert st["normal_map_baked"]
    axis = _box_edge_normals(dst)
    assert len(axis) > 50
    # smoothed across the edge, a corner normal leans 45 deg (0.71); kept hard it stays on its face
    assert np.percentile(axis, 5) > 0.95, f"hard edges were smoothed: p5 {np.percentile(axis, 5)}"


def test_lod_ladder_defaults_to_relative_tiers_for_a_static_mesh(two_material_crate, tmpd):
    out = os.path.join(tmpd, "lods")
    ladder = build_lod_ladder(two_material_crate, out)
    names = [lv["name"] for lv in ladder["levels"]]
    assert names == ["lod1", "lod2", "lod3"]
    src = ladder["source_triangles"]
    for lv, ratio in zip(ladder["levels"], (0.5, 0.25, 0.1)):
        assert lv["target_triangles"] == round(src * ratio)
        assert abs(lv["triangles"] - lv["target_triangles"]) < 0.2 * lv["target_triangles"]
        assert lv["texture_size"] is None
        assert _boundary_edges(lv["file"]) == 0, f"{lv['name']} tore at the material border"


def test_ladder_skips_a_rung_with_nothing_left_to_reduce(tmpd):
    V, F, UV, N = crate_sides(n=4)[0]                    # 18 triangles
    src = write_scene(os.path.join(tmpd, "tiny.glb"), [[_prim((V, F, UV, N))]])
    ladder = build_lod_ladder(src, os.path.join(tmpd, "tiny_lods"))
    assert ladder["levels"] == []
    # matched by message: test_decimate reloads the module, which re-creates the class
    with pytest.raises(ValueError, match="nothing to decimate") as exc:
        decimate_source_glb(src, os.path.join(tmpd, "tiny_out.glb"), 0.5)
    assert type(exc.value).__name__ == NothingToDecimate.__name__


def test_prop_profile_is_the_default(two_material_crate, tmpd):
    st = optimize_glb(two_material_crate, os.path.join(tmpd, "p.glb"), ratio=0.3, bake=False)
    assert st["head_boost"] == 1.0 and st["hand_boost"] == 1.0
    st = optimize_glb(two_material_crate, os.path.join(tmpd, "c.glb"), ratio=0.3, bake=False,
                      profile="character")
    assert st["head_boost"] > 1.0


# ---- buffer layouts exporters actually produce ---------------------------------------------

def test_primitives_sharing_one_vertex_array(sides, tmpd):
    a, b = merge(sides[:3]), merge(sides[3:])
    src = write_scene(os.path.join(tmpd, "shared.glb"), [[_prim(a, 0), _prim(b, 1)]],
                      materials=[{}, {}], share_position=True)
    g = GLTF2().load(src)
    p0, p1 = g.meshes[0].primitives
    assert p0.attributes.POSITION == p1.attributes.POSITION
    dst = os.path.join(tmpd, "shared_opt.glb")
    optimize_glb(src, dst, ratio=0.1)
    _, prims = read_prims(dst)
    for _, _, P, F, attrs, _ in prims:
        assert F.max() < len(P)
        assert all(len(v) == len(P) for v in attrs.values())
    assert _boundary_edges(dst) == 0


def test_extra_attributes_are_carried(sides, tmpd):
    """A custom attribute is an exact per-vertex copy in its own encoding; an extra UV set (a
    lightmap in TEXCOORD_2) is re-read off the surface like TEXCOORD_0. Before, both kept an index
    into the old accessor table and read whatever accessor landed there after the repack."""
    a = merge(sides[:3])
    tag = np.stack([np.arange(len(a[0])), np.arange(len(a[0])) * 2], 1).astype(np.float32)
    fid = (np.arange(len(a[0])) % 7).astype(np.uint16)
    lightmap = (a[0][:, :2] * 0.5 + 0.5).astype(np.float32)          # linear in position
    src = write_scene(os.path.join(tmpd, "extra.glb"), [[_prim(a)]],
                      extra_attrs={"_TAG": ([[tag]], 5126, "VEC2", False),
                                   "_FEATURE_ID_0": ([[fid]], UNSIGNED_SHORT, "SCALAR", False),
                                   "TEXCOORD_2": ([[lightmap]], 5126, "VEC2", False)})
    dst = os.path.join(tmpd, "extra_opt.glb")
    optimize_glb(src, dst, ratio=0.1)
    g, prims = read_prims(dst)
    _, _, P, F, attrs, _ = prims[0]
    assert all(len(v) == len(P) for v in attrs.values())
    # every _TAG row is an untouched copy of one source row, from a vertex nearby
    idx = attrs["_TAG"][:, 0].astype(np.int64)
    assert np.array_equal(attrs["_TAG"][:, 1], idx * 2.0)
    assert np.linalg.norm(a[0][idx] - P, axis=1).max() < 0.2
    acc = g.accessors[g.meshes[0].primitives[0].attributes._FEATURE_ID_0]
    assert acc.componentType == UNSIGNED_SHORT
    assert np.array_equal(attrs["_FEATURE_ID_0"], (idx % 7).astype(np.uint16))
    # the lightmap still matches the surface where the vertex now stands
    assert np.abs(attrs["TEXCOORD_2"] - (P[:, :2] * 0.5 + 0.5)).max() < 0.01


def test_interleaved_accessors_read_correctly(tmp_path):
    """A strided bufferView (POSITION and NORMAL interleaved) must read like a packed one."""
    from pygltflib import Accessor, BufferView, Buffer, FLOAT
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(1000, 3)).astype(np.float32)
    nrm = rng.normal(size=(1000, 3)).astype(np.float32)
    inter = np.concatenate([pos, nrm], axis=1).tobytes()
    g = GLTF2()
    g.bufferViews = [BufferView(buffer=0, byteOffset=0, byteLength=len(inter), byteStride=24)]
    g.accessors = [Accessor(bufferView=0, byteOffset=0, componentType=FLOAT, count=1000,
                            type="VEC3"),
                   Accessor(bufferView=0, byteOffset=12, componentType=FLOAT, count=1000,
                            type="VEC3")]
    g.buffers = [Buffer(byteLength=len(inter))]
    assert np.array_equal(_acc(g, inter, 0), pos)
    assert np.array_equal(_acc(g, inter, 1), nrm)


def test_compressed_geometry_is_refused_not_misread(two_material_crate, tmpd):
    g = GLTF2().load(two_material_crate)
    g.extensionsUsed = ["EXT_meshopt_compression"]
    src = os.path.join(tmpd, "meshopt.glb")
    g.save(src)
    with pytest.raises(ValueError, match="EXT_meshopt_compression"):
        optimize_glb(src, os.path.join(tmpd, "meshopt_out.glb"), ratio=0.5)


# ---- the bake on environment layouts --------------------------------------------------------

def _image_of(g, tex_idx):
    blob = g.binary_blob()
    img = g.images[g.textures[tex_idx].source]
    bv = g.bufferViews[img.bufferView]
    raw = bytes(blob[bv.byteOffset:bv.byteOffset + bv.byteLength])
    return raw, np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def test_tiling_uvs_are_not_baked(tmpd):
    t = merge(crate_sides(n=40, tiling=True))
    nmap = flat_normal_map(64)
    nmap[::8] = (200, 128, 200)                           # recognisable stripes
    src = write_scene(os.path.join(tmpd, "tiling.glb"), [[_prim(t)]], materials=[{"normal": 0}],
                      images=[nmap])
    dst = os.path.join(tmpd, "tiling_opt.glb")
    st = optimize_glb(src, dst, ratio=0.1)
    assert not st["normal_map_baked"]
    assert "tiling" in st["bake_skipped"][0]["skipped"]
    g = GLTF2().load(dst)
    _, img = _image_of(g, g.materials[0].normalTexture.index)
    assert np.array_equal(img, nmap), "the shared tiling normal map was rewritten"
    _, prims = read_prims(dst)
    assert "TANGENT" not in prims[0][4] or len(prims[0][4]["TANGENT"]) == len(prims[0][2])


def test_overlapping_uvs_are_not_baked(sides, tmpd):
    # all six sides stacked on the SAME atlas cell (a common trick for identical panels)
    parts = [(V, F, (UV - UV.min(0)) / (UV.max(0) - UV.min(0)), N) for V, F, UV, N in sides]
    src = write_scene(os.path.join(tmpd, "stacked.glb"), [[_prim(merge(parts))]])
    st = optimize_glb(src, os.path.join(tmpd, "stacked_opt.glb"), ratio=0.1)
    assert not st["normal_map_baked"]
    assert "overlapping" in st["bake_skipped"][0]["skipped"]


def test_flat_floor_is_baked_by_rays_not_fallbacks(tmpd):
    src = write_scene(os.path.join(tmpd, "floor.glb"), [[_prim(floor_tile())]])
    st = optimize_glb(src, os.path.join(tmpd, "floor_opt.glb"), ratio=0.05)
    assert st["normal_map_baked"]
    assert st["bake"][0]["cage_hit_rate"] > 0.9


def test_meshes_sharing_a_material_share_one_baked_map(sides, tmpd):
    a, b = merge(sides[:3]), merge(sides[3:])           # disjoint halves of one atlas
    src = write_scene(os.path.join(tmpd, "twomesh.glb"), [[_prim(a)], [_prim(b)]],
                      nodes=[(0, (0, 0, 0)), (1, (3, 0, 0))])
    dst = os.path.join(tmpd, "twomesh_opt.glb")
    st = optimize_glb(src, dst, ratio=0.1)
    assert st["normal_map_baked"] and len(st["bake"]) == 2
    g = GLTF2().load(dst)
    assert len(g.materials) == 1 and len(g.images) == 1
    _, img = _image_of(g, g.materials[0].normalTexture.index)
    # the atlas is 3x2 cells: the first mesh owns the top row, the second the bottom row. Both
    # rows must carry baked relief, not one row baked and the other left flat.
    h = img.shape[0]
    for row in (img[: h // 2], img[h // 2:]):
        assert np.abs(row[..., :2].astype(int) - 128).mean() > 1.0


def test_a_shared_material_is_cloned_not_hijacked(sides, tmpd):
    """A material also used by a primitive that is NOT decimated (too small) must keep its own
    normal map for that primitive; the decimated one gets a clone."""
    nmap = flat_normal_map(64)
    big = merge(sides[:3])
    small = crate_sides(n=4)[3]                            # 18 triangles: left alone
    src = write_scene(os.path.join(tmpd, "sharedmat.glb"), [[_prim(big)], [_prim(small)]],
                      materials=[{"normal": 0}], images=[nmap])
    dst = os.path.join(tmpd, "sharedmat_opt.glb")
    st = optimize_glb(src, dst, ratio=0.1)
    assert st["normal_map_baked"]
    g = GLTF2().load(dst)
    m_big = g.meshes[0].primitives[0].material
    m_small = g.meshes[1].primitives[0].material
    assert m_small == 0 and m_big != m_small
    _, img = _image_of(g, g.materials[m_small].normalTexture.index)
    assert np.array_equal(img, nmap), "the untouched primitive's normal map was replaced"
    assert g.materials[m_big].normalTexture.index != g.materials[m_small].normalTexture.index


# ---- reporting ------------------------------------------------------------------------------

def test_stats_and_world_space_loader(two_material_crate, tmpd):
    st = mesh_stats(two_material_crate)
    assert st["primitives"] == 2 and st["materials"] == 2 and st["optimizable"]
    a = merge(crate_sides(n=10)[:1])
    src = write_scene(os.path.join(tmpd, "placed.glb"), [[_prim(a)]], nodes=[(0, (5, 0, 0))])
    _, _, prims = load_triangles(src)
    assert np.allclose(prims[0]["P"].mean(0)[0], a[0].mean(0)[0] + 5, atol=1e-5)


# ---- seams on flat faces --------------------------------------------------------------------

def _two_chart_panel(n=61):
    """A flat panel whose left and right halves live in two different atlas charts, split along
    x = 0 — the layout of a wall with its decal in its own chart. Geometrically the seam costs
    nothing, which is exactly what let a collapse run across it."""
    from _prop_fixtures import _grid
    uv01, _, F = _grid(n, relief=0.0)
    V = np.stack([uv01[:, 0] * 2 - 1, uv01[:, 1] * 2 - 1, np.zeros(len(uv01))], 1)
    left = V[:, 0] <= 0
    # split the seam column: the right chart gets its own copies of the x == 0 vertices
    seam = np.isclose(V[:, 0], 0)
    dup = np.flatnonzero(seam)
    remap = np.arange(len(V))
    remap[dup] = len(V) + np.arange(len(dup))
    Vb = np.concatenate([V, V[dup]])
    cent = V[F].mean(1)
    Fb = np.where((cent[:, 0] > 0)[:, None], remap[F], F)
    expect_uv = lambda P, right: np.stack([np.where(right, 0.55 + 0.45 * P[:, 0],
                                                    0.45 * (P[:, 0] + 1)), (P[:, 1] + 1) / 2], 1)
    right = np.r_[~left, np.ones(len(dup), bool)]
    UV = expect_uv(Vb, right)
    N = np.tile([0.0, 0.0, 1.0], (len(Vb), 1))
    return (Vb.astype(np.float32), Fb.astype(np.int64), UV.astype(np.float32),
            N.astype(np.float32)), expect_uv


def test_collapses_never_cross_a_uv_seam_on_a_flat_face(tmpd):
    part, expect_uv = _two_chart_panel()
    src = write_scene(os.path.join(tmpd, "panel.glb"), [[_prim(part)]])
    dst = os.path.join(tmpd, "panel_opt.glb")
    st = optimize_glb(src, dst, ratio=0.1, bake=False)
    assert st["seam_lock"] and not st["seam_limited"]
    assert st["triangles_after"] <= 1.1 * st["target_triangles"]
    _, prims = read_prims(dst)
    _, _, P, F, attrs, _ = prims[0]
    uv = attrs["TEXCOORD_0"]
    # every face lies wholly in one chart, and its UVs are where that chart says they should be
    right = uv[:, 0] > 0.5
    assert (right[F].all(1) | (~right[F]).all(1)).all(), "a face spans both charts"
    assert np.abs(uv - expect_uv(P, right)).max() < 1e-3


def test_baked_map_keeps_a_lossy_webp_pipeline_lossy(sides, tmpd):
    from retopotool.bake_normals import encode_map
    rgb = np.full((64, 64, 3), 128, np.uint8)
    for mode in (False, True):
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="WEBP", lossless=mode)
        data, mime = encode_map(rgb, buf.getvalue())
        assert mime == "image/webp" and (data[12:16] == b"VP8L") == mode
    data, mime = encode_map(rgb, None)
    assert mime == "image/png"
