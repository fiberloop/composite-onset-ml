"""Reading the data files (format: docs/data.md)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr

COMPONENTS = ("e11", "e33", "g13")


def data_dir() -> Path:
    """The folder of the data files: $ONSETML_DATA if set, else data/ of the repository."""
    env = os.environ.get("ONSETML_DATA")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "data"


def path_of(test: str | Path) -> Path:
    """The file of a test, given by its name (looked up in data_dir()) or by a path."""
    p = Path(test)
    if p.suffix == ".h5" or p.exists():
        return p
    return data_dir() / f"{test}.h5"


def load(test: str | Path) -> xr.Dataset:
    """A test as an xarray.Dataset, read lazily (arrays come from disk when used)."""
    path = path_of(test)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: put the file in data/ or set ONSETML_DATA")
    return xr.open_dataset(path, engine="netcdf4")


def history(ds: xr.Dataset, element: int) -> tuple[np.ndarray, np.ndarray]:
    """The load (n_image,) [kN] and the strain (n_image, 3) of one element."""
    return ds["load_kN"].values, ds["strain"].isel(element=element).values


def field(ds: xr.Dataset, image: int) -> np.ndarray:
    """The strain of every element (n_element, 3) at one image (its number, not its position)."""
    return ds["strain"].sel(image=image).values


def usable(ds: xr.Dataset) -> np.ndarray:
    """Per image (n_image,): converged and not blurred by a camera."""
    keep = ~ds["mask_not_converged"].values
    if "mask_blurred" in ds:
        keep &= ~ds["mask_blurred"].values
    return keep


def loading_images(ds: xr.Dataset, include_static: bool = False) -> np.ndarray:
    """The image numbers of the loading, up to the peak: neither the reference image nor, unless
    asked, the static images; not after the peak; only usable ones (converged, not blurred)."""
    keep = ~ds["mask_reference"].values & ~ds["mask_after_peak"].values & usable(ds)
    if not include_static:
        keep &= ~ds["mask_static"].values
    return ds["image"].values[keep]


def static_noise(ds: xr.Dataset) -> np.ndarray:
    """The standard deviation of each element's strain over the usable static images, the
    reference image left out (its strain is zero by definition) (n_element, 3): the noise of the
    measurement at rest."""
    keep = ds["mask_static"].values & ~ds["mask_reference"].values & usable(ds)
    return ds["strain"].sel(image=ds["image"].values[keep]).std("image", ddof=1).values
