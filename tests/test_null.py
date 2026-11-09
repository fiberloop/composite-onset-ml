"""The null model on synthetic series (the tests never touch the real data)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from scipy import stats

from onsetml import data, null


def test_fit_lines_and_departure_recover_a_line() -> None:
    f = np.linspace(0, 1, 50)
    k = np.array([[1e-3, 2e-3, -3e-3]])
    b = np.array([[1e-5, 0.0, 2e-5]])
    X = f[:, None, None] * k[None] + b[None]
    X[f > 0.6] += 5e-4  # a departure after the window
    kk, bb = null.fit_lines(f, X)
    assert np.allclose(kk, k) and np.allclose(bb, b)
    eta = null.departure(f, X)
    assert np.allclose(eta[f <= 0.6], 0) and np.allclose(eta[f > 0.6], 5e-4)
    with pytest.raises(ValueError):
        null.fit_lines(f[:2], X[:2])


def test_sigma_under_load_is_the_spread_around_the_line() -> None:
    rng = np.random.default_rng(0)
    f = np.linspace(0, 1, 4000)
    X = 1e-3 * f[:, None, None] + rng.normal(0, 7e-5, (4000, 1, 3))
    assert np.allclose(null.sigma_under_load(f, X), 7e-5, rtol=0.1)


def _static_ds(K: int, cov: np.ndarray, n_el: int, seed: int = 0) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    S = rng.multivariate_normal(np.zeros(3), cov, size=(K + 1, n_el))  # image 0: reference
    S[0] = 0
    images = np.arange(K + 1)
    return xr.Dataset(
        {"strain": (("image", "element", "component"), S),
         "mask_reference": ("image", images == 0),
         "mask_static": ("image", np.ones(K + 1, bool)),
         "mask_not_converged": ("image", np.zeros(K + 1, bool))},
        coords={"image": images, "component": list(data.COMPONENTS)})


def test_static_covariance_diagonal_is_the_static_noise() -> None:
    cov = np.diag([1.0, 2.0, 4.0]) * 1e-8
    ds = _static_ds(40, cov, 5)
    S, K = null.static_covariance(ds)
    assert K == 40 and S.shape == (5, 3, 3)
    assert np.allclose(np.sqrt(np.einsum("eii->ei", S)), data.static_noise(ds))


def test_psi_with_a_diagonal_sigma_is_the_root_sum_of_squares() -> None:
    rng = np.random.default_rng(1)
    eta = rng.normal(size=(7, 4, 3))
    sd = np.array([1.0, 2.0, 3.0])
    cov = np.broadcast_to(np.diag(sd**2), (4, 3, 3))
    assert np.allclose(null.psi(eta, cov), np.sqrt(((eta / sd) ** 2).sum(axis=2)))


def test_hotelling_law_of_psi_with_an_estimated_sigma() -> None:
    # Sigma from K = 20 images, psi of a new zero-mean observation: psi^2 ~ T^2(3, 19)
    rng = np.random.default_rng(2)
    K, n = 20, 20000
    cov = np.array([[1.0, 0.3, 0.0], [0.3, 2.0, 0.5], [0.0, 0.5, 1.5]])
    S = rng.multivariate_normal(np.zeros(3), cov, size=(K, n))
    d = S - S.mean(axis=0)
    est = np.einsum("kei,kej->eij", d, d) / (K - 1)
    new = rng.multivariate_normal(np.zeros(3), cov, size=(1, n))
    p = null.psi(new, est)[0]
    thr = null.hotelling_threshold(0.01, nu=K - 1, c=1.0 + 1.0 / K)  # the mean was estimated too
    new_c = new - S.mean(axis=0)[None]
    p_c = null.psi(new_c, est)[0]
    assert abs((p_c > thr).mean() - 0.01) < 0.003
    assert np.isclose(null.hotelling_sf(thr, nu=K - 1, c=1.0 + 1.0 / K), 0.01)
    # with many images T^2 tends to chi_3
    assert np.isclose(null.hotelling_threshold(1e-3, nu=10**6), stats.chi.isf(1e-3, 3), rtol=1e-3)
    assert (p > null.hotelling_threshold(0.01, nu=K - 1)).mean() < 0.02
    with pytest.raises(ValueError):
        null.hotelling_threshold(0.01, nu=2)


def test_student_threshold() -> None:
    assert np.isclose(null.student_threshold(0.05, 10**6), 1.96, atol=1e-3)
    assert null.student_threshold(0.05, 19) > 2.0


def test_cusum_alarms_after_a_step_and_not_before() -> None:
    f = np.linspace(0, 1, 101)
    z = np.zeros((101, 2))
    z[f > 0.6, 0] = 4.0  # +2 per image beyond kappa: S > 10 at the 6th image after 0.6
    alarm = null.cusum(f, z, kappa=2.0, h=10.0)
    assert alarm[1] == -1
    assert f[alarm[0]] == pytest.approx(f[np.argmax(f > 0.6)] + 5 * 0.01)
