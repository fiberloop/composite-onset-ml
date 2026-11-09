"""Drawing a field on the mesh of the face: per element (flat) or averaged at the nodes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.tri import Triangulation


def triangulation(ds: xr.Dataset) -> Triangulation:
    """The mesh of the face as a matplotlib Triangulation (face axes x1, x3, in mm)."""
    return Triangulation(ds["node_x1"].values, ds["node_x3"].values, ds["elements"].values)


def nodal_average(ds: xr.Dataset, values: np.ndarray) -> np.ndarray:
    """Element values (n_element, ...) averaged at the nodes (n_node, ...).

    Each node takes the mean of the elements around it, weighted by their area; NaN elements
    are left out, and a node with no finite element around it is NaN. This is what FE and DIC
    software show by default. It smooths: a node mixes about six triangles here.
    """
    v = np.asarray(values, dtype=float)
    conn = ds["elements"].values
    flat = v.reshape(v.shape[0], -1)
    x = np.stack([ds["node_x1"].values[conn], ds["node_x3"].values[conn]], axis=-1)
    d1, d2 = x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]
    area = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    ok = np.isfinite(flat)
    w = area[:, None] * ok
    num = np.zeros((ds.sizes["node"], flat.shape[1]))
    den = np.zeros_like(num)
    for k in range(conn.shape[1]):
        np.add.at(num, conn[:, k], w * np.where(ok, flat, 0.0))
        np.add.at(den, conn[:, k], w)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return out.reshape((ds.sizes["node"],) + v.shape[1:])


def element_map(
    ds: xr.Dataset,
    values: np.ndarray,
    ax: Axes | None = None,
    label: str = "",
    nodal: bool = False,
    colorbar: bool = True,
    **kwargs,
):
    """One value per element (n_element,) drawn on the mesh; NaN left blank.

    nodal=False (default): flat on each triangle, the values exactly as stored.
    nodal=True: averaged at the nodes (`nodal_average`) and interpolated across each triangle,
    the smooth picture of FE and DIC software.
    Returns the axes (the drawn collection is `ax.collections[-1]`).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    tri = triangulation(ds)
    v = np.asarray(values, dtype=float)
    if nodal:
        n = nodal_average(ds, v)
        tri.set_mask(~np.isfinite(n[tri.triangles]).all(axis=1))
        pc = ax.tripcolor(tri, np.nan_to_num(n), shading="gouraud", **kwargs)
    else:
        pc = ax.tripcolor(tri, facecolors=v, **kwargs)
    ax.set_aspect("equal")
    ax.set_xlabel("x1 [mm]")
    ax.set_ylabel("x3 [mm]")
    if colorbar:
        ax.figure.colorbar(pc, ax=ax, label=label, shrink=0.8)
    return ax
