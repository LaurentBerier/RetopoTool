"""Small synthetic meshes for the pipeline / CLI tests (the decimation suite builds its own)."""
import os

import pytest

from _fixtures import _bumpy_sphere, _write_glb, _write_skinned_glb


@pytest.fixture(scope="session")
def small_sphere():
    """~19.6k triangles: big enough to decimate meaningfully, small enough to keep the suite fast."""
    return _bumpy_sphere(n_theta=100, n_phi=100)


@pytest.fixture(scope="session")
def small_mesh_glb(small_sphere, tmp_path_factory):
    V, F, UV, N = small_sphere
    return _write_glb(os.path.join(tmp_path_factory.mktemp("mesh"), "mesh.glb"), V, F, UV, N)


@pytest.fixture(scope="session")
def small_rig_glb(small_sphere, tmp_path_factory):
    V, F, UV, N = small_sphere
    return _write_skinned_glb(os.path.join(tmp_path_factory.mktemp("rig"), "rig.glb"),
                              V, F, UV, N, orphans=2)
