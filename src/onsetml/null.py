"""The null model of the kickoff deck: one line per element and component, the departure from it
in noise units, its laws under H0, and a first sequential detector.

    from onsetml import data, null

    ds = data.load("ARC-00_m45_s2")
    images = data.loading_images(ds)
    f = ds["load_over_max"].sel(image=images).values            # (n_image,)
    X = ds["strain"].sel(image=images).values                   # (n_image, n_element, 3)
    eta = null.departure(f, X)                                  # strain minus its line
    z = eta / data.static_noise(ds)[None]                       # one component at a time
    S, K = null.static_covariance(ds)                           # 3x3 noise per element, K images
    psi = null.psi(eta, S)                                      # the three together
    thr = null.hotelling_threshold(1e-3, nu=K - 1)              # psi beyond it: alpha = 1e-3

Shapes: images first, then elements, then components, as in the file.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import stats

from onsetml import data

LOW = 0.30  # the fit window of the deck: F <= 0.3 F_max


def fit_lines(f: np.ndarray, X: np.ndarray, low: float = LOW) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares line of each element and component against the load over F <= low.

    f: (n_image,) load over its peak; X: (n_image, ...) strain. Returns the slope and the
    intercept, each of shape X.shape[1:].
    """
    w = f <= low
    if w.sum() < 3:
        raise ValueError(f"only {w.sum()} images with F <= {low}: too few to fit a line")
    A = np.column_stack([f[w], np.ones(w.sum())])
    coef, *_ = np.linalg.lstsq(A, X[w].reshape(w.sum(), -1), rcond=None)
    shape = X.shape[1:]
    return coef[0].reshape(shape), coef[1].reshape(shape)


def departure(f: np.ndarray, X: np.ndarray, low: float = LOW) -> np.ndarray:
    """eta: the strain minus its line fitted over F <= low, same shape as X."""
    k, b = fit_lines(f, X, low)
    return X - (f.reshape((-1,) + (1,) * (X.ndim - 1)) * k[None] + b[None])


def sigma_under_load(f: np.ndarray, X: np.ndarray, low: float = LOW) -> np.ndarray:
    """The spread of the residual of the line over F <= low (n - 2 degrees of freedom), per
    element and component: the noise under load, where the line holds. Where the line already
    bends inside the window, the bending counts as noise (deck, backup on sigma under load)."""
    w = f <= low
    eta = departure(f, X, low)[w]
    return np.sqrt((eta**2).sum(axis=0) / (w.sum() - 2))


def static_covariance(ds: xr.Dataset) -> tuple[np.ndarray, int]:
    """Sigma_e: the 3x3 covariance of each element's strain over the usable static images, the
    reference image left out (n_element, 3, 3), and the number K of images it is estimated from.
    Its diagonal is data.static_noise(ds) squared."""
    keep = ds["mask_static"].values & ~ds["mask_reference"].values & data.usable(ds)
    S = ds["strain"].sel(image=ds["image"].values[keep]).values
    d = S - S.mean(axis=0)
    return np.einsum("kei,kej->eij", d, d) / (S.shape[0] - 1), int(S.shape[0])


def psi(eta: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """psi = sqrt(eta^T Sigma^-1 eta) per image and element: the departure of the three
    components together, in noise units. eta: (n_image, n_element, 3); cov: (n_element, 3, 3).
    With Sigma known it follows chi_3 under H0; with Sigma estimated, Hotelling T^2 (below)."""
    return np.sqrt(np.einsum("nei,eij,nej->ne", eta, np.linalg.inv(cov), eta))


def _hotelling_scale(nu: int, p: int, c: float) -> tuple[float, int, int]:
    if nu < p:
        raise ValueError(f"nu = {nu} < p = {p}: Sigma cannot be inverted")
    return (nu - p + 1) / (p * nu * c), p, nu - p + 1


def hotelling_sf(x: np.ndarray, nu: int, p: int = 3, c: float = 1.0) -> np.ndarray:
    """P(psi > x) under H0 when Sigma is estimated with nu degrees of freedom (K - 1 for K
    images): psi^2 / c ~ T^2(p, nu), and T^2 (nu - p + 1) / (p nu) ~ F(p, nu - p + 1).
    c = 1 + 1/n when the mean is also estimated from n images (held-out static images)."""
    g, d1, d2 = _hotelling_scale(nu, p, c)
    return stats.f.sf(np.asarray(x, dtype=float) ** 2 * g, d1, d2)


def hotelling_threshold(alpha: float, nu: int, p: int = 3, c: float = 1.0) -> float:
    """The threshold on psi with false-alarm rate alpha per element and image under H0: the
    1 - alpha quantile of its Hotelling law. Choose alpha, then the threshold (deck, backup on
    the threshold); under load, calibrate it on loaded images too, the noise there is larger."""
    g, d1, d2 = _hotelling_scale(nu, p, c)
    return float(np.sqrt(stats.f.isf(alpha, d1, d2) / g))


def student_threshold(alpha: float, nu: int) -> float:
    """The same for one component: |z| beyond it with rate alpha when sigma is estimated with
    nu degrees of freedom, z ~ Student t_nu."""
    return float(stats.t.isf(alpha / 2, nu))


def cusum(f: np.ndarray, z: np.ndarray, kappa: float = 2.0, h: float = 10.0,
          low: float = LOW) -> np.ndarray:
    """One-sided CUSUM of |z| started at the end of the fit window: S_n = max(0, S_{n-1} +
    |z_n| - kappa) for F > low, 0 before. Returns the index of the first image with S_n > h for
    each series (z: (n_image, ...)), -1 if none. kappa and h are a first guess (deck: 2 and
    10), to be calibrated on images with a known truth."""
    S = np.zeros(z.shape[1:])
    alarm = np.full(z.shape[1:], -1)
    for n in range(z.shape[0]):
        if f[n] > low:
            S = np.maximum(0.0, S + np.abs(z[n]) - kappa)
        hit = (S > h) & (alarm < 0)
        alarm[hit] = n
    return alarm
