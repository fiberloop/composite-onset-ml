"""A first look at one test: the load curve, the strain at the peak, a few element histories and
the noise at rest. Writes outputs/first_look_<test>_*.png.

    python scripts/first_look.py ARC-00_m45_s2
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from onsetml import data, plotting


def main(test: str) -> None:
    ds = data.load(test)
    out = Path(__file__).resolve().parents[1] / "outputs"
    out.mkdir(exist_ok=True)
    peak = int(ds.attrs["peak_image"])
    print(
        f"{test}: {ds.sizes['image']} images, {ds.sizes['element']} elements, "
        f"peak {ds.attrs['peak_load_kN']:.3f} kN at image {peak}"
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ds["stroke_mm"], ds["load_kN"])
    ax.set_xlabel("stroke [mm]")
    ax.set_ylabel("load [kN]")
    fig.savefig(out / f"first_look_{test}_load.png", dpi=120, bbox_inches="tight")

    fig, axes = plt.subplots(3, 1, figsize=(9, 10))
    e = data.field(ds, peak)
    for c, name in enumerate(data.COMPONENTS):
        lim = np.nanpercentile(np.abs(e[:, c]), 99)
        plotting.element_map(
            ds,
            e[:, c],
            ax=axes[c],
            label=f"{name} at image {peak}",
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
        )
    fig.savefig(out / f"first_look_{test}_peak.png", dpi=120, bbox_inches="tight")

    # a few elements: the nearest the notch root, mid-way, and far from it
    d = ds["element_root_mm"].values
    picks = [int(np.argmin(d)), int(np.argmin(np.abs(d - 2.0))), int(np.argmax(d))]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for c, name in enumerate(data.COMPONENTS):
        for el in picks:
            load, strain = data.history(ds, el)
            axes[c].plot(
                strain[:, c] * 1e6, load, label=f"element {el}, {d[el]:.2f} mm from a root"
            )
        axes[c].set_xlabel(f"{name} [microstrain]")
    axes[0].set_ylabel("load [kN]")
    axes[0].legend(fontsize=8)
    fig.savefig(out / f"first_look_{test}_histories.png", dpi=120, bbox_inches="tight")

    sd = data.static_noise(ds)
    print(
        "noise at rest, median over the elements [microstrain]:",
        dict(zip(data.COMPONENTS, np.round(np.nanmedian(sd, axis=0) * 1e6).tolist(), strict=True)),
    )
    print(f"figures in {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ARC-00_m45_s2")
