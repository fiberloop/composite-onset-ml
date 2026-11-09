"""The reader on a small synthetic file written here (the tests never touch the real data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from onsetml import data, plotting


@pytest.fixture
def small(tmp_path: Path) -> Path:
    """Two triangles, five images: 0 the reference, 1-2 static, 3-4 loading (peak at 4)."""
    n_img = 5
    rng = np.random.default_rng(0)
    strain = rng.normal(0, 1e-4, (n_img, 2, 3))
    strain[3:] += np.array([1e-3, 2e-3, 3e-3])
    images = np.arange(n_img)
    ds = xr.Dataset(
        {
            "strain": (("image", "element", "component"), strain.astype(np.float32)),
            "displacement": (("image", "node", "direction"), np.zeros((n_img, 4, 3), np.float32)),
            "load_kN": ("image", np.array([0.0, 0.0, 0.0, 0.5, 1.0])),
            "node_x1": ("node", np.array([0.0, 1.0, 1.0, 0.0])),
            "node_x3": ("node", np.array([0.0, 0.0, 1.0, 1.0])),
            "elements": (("element", "corner"), np.array([[0, 1, 2], [0, 2, 3]])),
            "mask_reference": ("image", images == 0),
            "mask_static": ("image", np.isin(images, [1, 2])),
            "mask_not_converged": ("image", np.zeros(n_img, bool)),
            "mask_after_peak": ("image", np.zeros(n_img, bool)),
        },
        coords={"image": images, "component": list(data.COMPONENTS)},
        attrs={"peak_image": 4},
    )
    path = tmp_path / "SYN.h5"
    ds.to_netcdf(path, engine="netcdf4")
    return path


def test_load_by_name_and_by_path(small: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONSETML_DATA", str(small.parent))
    assert data.load("SYN").sizes["element"] == 2
    assert data.load(small).sizes["image"] == 5
    with pytest.raises(FileNotFoundError):
        data.load("NONE")


def test_history_field_and_images(small: Path) -> None:
    ds = data.load(small)
    load, strain = data.history(ds, 1)
    assert load.shape == (5,) and strain.shape == (5, 3)
    assert np.allclose(data.field(ds, 4), ds["strain"].values[4])
    assert data.loading_images(ds).tolist() == [3, 4]
    assert data.loading_images(ds, include_static=True).tolist() == [1, 2, 3, 4]
    sd = data.static_noise(ds)
    assert sd.shape == (2, 3) and np.all(sd > 0)


def test_element_map_draws_one_value_per_triangle(small: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    ax = plotting.element_map(data.load(small), np.array([1.0, np.nan]))
    assert len(ax.collections) == 1


def test_nodal_average_weighs_the_elements_around_each_node(small: Path) -> None:
    ds = data.load(small)
    assert np.allclose(plotting.nodal_average(ds, np.array([1.0, 3.0])), [2, 1, 2, 3])
    n = plotting.nodal_average(ds, np.array([1.0, np.nan]))
    assert np.allclose(n[:3], 1) and np.isnan(n[3])
    assert plotting.nodal_average(ds, np.ones((2, 3))).shape == (4, 3)

    import matplotlib

    matplotlib.use("Agg")
    ax = plotting.element_map(ds, np.array([1.0, np.nan]), nodal=True)
    assert len(ax.collections) == 1


def test_blurred_and_reference_images_are_left_out(small: Path) -> None:
    ds = data.load(small).load()
    ds["mask_static"] = ("image", np.isin(ds["image"].values, [0, 1, 2]))
    ds["mask_blurred"] = ("image", ds["image"].values == 3)
    assert data.loading_images(ds).tolist() == [4]
    one = data.static_noise(ds)
    expected = ds["strain"].sel(image=[1, 2]).std("image", ddof=1).values
    assert np.allclose(one, expected)
