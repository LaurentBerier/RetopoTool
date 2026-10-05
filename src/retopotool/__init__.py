"""retopotool — seam-preserving GLB decimation, normal-map bake and skin-preserving LOD ladders.

    from retopotool import optimize_glb, build_lod_ladder, mesh_stats

    mesh_stats("character.glb")                              # cheap triangle/vertex/skin report
    optimize_glb("character.glb", "character_opt.glb")       # unrigged: ~400k tris + normal bake
    build_lod_ladder("character_rigged.glb", "lods/")        # rigged: high/medium/low/minimum
"""
from . import mesh_quality
from .bake_normals import bake_normal_map
from .decimate import (DENSE_TRIANGLE_THRESHOLD, TARGET_TRIANGLES, build_skin_proxy,
                       decimate_rigged_glb, decimate_source_glb, mesh_stats)
from .pipeline import (DEFAULT_LOD_TIERS, LodTier, build_lod_ladder, merge_lod_levels,
                       optimize_glb, warn_lod_quality)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_LOD_TIERS",
    "DENSE_TRIANGLE_THRESHOLD",
    "LodTier",
    "TARGET_TRIANGLES",
    "bake_normal_map",
    "build_lod_ladder",
    "build_skin_proxy",
    "decimate_rigged_glb",
    "decimate_source_glb",
    "merge_lod_levels",
    "mesh_quality",
    "mesh_stats",
    "optimize_glb",
    "warn_lod_quality",
]
