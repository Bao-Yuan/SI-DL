from __future__ import annotations

import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, matrix_rank
from scipy.optimize import differential_evolution
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:
    lowess = None


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.usetex"] = False
PLOT_LABEL_FONTSIZE = 17
PLOT_TICK_FONTSIZE = 13
PLOT_TITLE_FONTSIZE = 17
PLOT_COLORBAR_FONTSIZE = 13

DEFAULT_GAUSSIAN_BANDWIDTH = 0.03
DEFAULT_BANDWIDTH_GRID = np.linspace(0.03, 0.30, 10)
DEFAULT_KERNEL_RADIUS_SCALE = 5.0
DEFAULT_MIRROR_MAX_DIM = 2
DEFAULT_LOWESS_FRAC = 2.0 / 3.0
DEFAULT_LOWESS_FRAC_GRID = np.linspace(0.10, 0.80, 8)
DEFAULT_LOWESS_IT = 3
DEFAULT_LOWESS_DELTA = 0.0


def calc_basis(D_in: np.ndarray, col_range: int) -> np.matrix:
    """
    Compute Buckingham-pi basis vectors using the same convention as IT_PI.

    D_in has one column per dimensional input variable and one row per base
    dimension. Each dimensionless exponent vector omega satisfies

        D_in @ omega = 0.

    This implementation assumes the first rank(D_in) columns form an
    invertible dimensional core, matching the notebooks in IT_PI-main.
    """
    D_in = np.asmatrix(D_in, dtype=float)
    num_rows = np.shape(D_in)[0]
    Din1, Din2 = D_in[:, :num_rows], D_in[:, num_rows:]
    basis_matrices = []
    for i in range(col_range):
        x2 = np.zeros((col_range, 1))
        x2[i, 0] = -1.0
        x1 = -inv(Din1) * Din2 * x2
        basis_matrices.append(np.vstack((x1, x2)))
    return np.asmatrix(np.array(basis_matrices).reshape(col_range, -1))


def calc_pi_omega(coef_pi: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Construct Pi = product_j X_j ** coef_j for positive dimensional data X."""
    coef_pi = np.asarray(coef_pi, dtype=float)
    if coef_pi.ndim == 1:
        coef_pi = coef_pi.reshape(1, -1)
    X = np.asarray(X, dtype=float)
    if np.any(X <= 0.0):
        raise ValueError("All dimensional inputs must be positive to form power-law Pi groups.")
    return np.exp(np.log(X) @ coef_pi.T)


def calc_log_pi_omega(coef_pi: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Construct log(Pi) = sum_j coef_j log(X_j)."""
    coef_pi = np.asarray(coef_pi, dtype=float)
    if coef_pi.ndim == 1:
        coef_pi = coef_pi.reshape(1, -1)
    X = np.asarray(X, dtype=float)
    if np.any(X <= 0.0):
        raise ValueError("All dimensional inputs must be positive to form power-law Pi groups.")
    return np.log(X) @ coef_pi.T


def create_labels(omega: np.ndarray, variables: list[str]) -> list[str]:
    """Create IT_PI-style labels for power-law dimensionless groups."""
    labels = []
    omega = np.asarray(omega, dtype=float)
    if omega.ndim == 1:
        omega = omega.reshape(1, -1)

    for row in omega:
        positive_part = ""
        negative_part = ""
        for i, value in enumerate(row):
            value = float(np.round(value, 2))
            if abs(value) < 1e-12:
                continue
            term = f"{variables[i]}^{{{abs(value):.2g}}}"
            if value > 0:
                positive_part = term if positive_part == "" else positive_part + f" \\cdot {term}"
            else:
                negative_part = term if negative_part == "" else negative_part + f" \\cdot {term}"
        if negative_part == "":
            labels.append(f"${positive_part}$")
        elif positive_part == "":
            labels.append(f"$\\frac{{1}}{{{negative_part}}}$")
        else:
            labels.append(f"$\\frac{{{positive_part}}}{{{negative_part}}}$")
    return labels


def _as_2d_u(U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    if U.ndim != 2:
        raise ValueError("U must have shape (n,) or (n, d).")
    return U


def mirror_augmented_samples(
    U: np.ndarray,
    max_dim: int | None = DEFAULT_MIRROR_MAX_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add reflected copies across the observed rectangular boundary of U.

    For two-dimensional U, this returns 9 copies per sample: the original
    points, four edge reflections, and four corner reflections. Higher
    dimensions are disabled by default because the augmented size is n * 3**d.
    Pass max_dim=None to opt into that cost explicitly.
    """
    U = _as_2d_u(U)
    n, d = U.shape
    if max_dim is not None and d > max_dim:
        raise ValueError(
            f"mirror boundary is limited to d <= {max_dim} by default; "
            "pass max_dim=None, or mirror_max_dim=None in the estimator, "
            "to allow n * 3**d augmented samples."
        )
    lower = np.min(U, axis=0)
    upper = np.max(U, axis=0)
    spans = upper - lower

    transforms: list[tuple[int, ...]] = [()]
    for _ in range(d):
        transforms = [prefix + (choice,) for prefix in transforms for choice in (0, -1, 1)]

    augmented = []
    source = []
    for transform in transforms:
        U_ref = U.copy()
        for j, choice in enumerate(transform):
            if spans[j] <= 0.0 or choice == 0:
                continue
            if choice < 0:
                U_ref[:, j] = 2.0 * lower[j] - U_ref[:, j]
            else:
                U_ref[:, j] = 2.0 * upper[j] - U_ref[:, j]
        augmented.append(U_ref)
        source.append(np.arange(n))

    return np.vstack(augmented), np.concatenate(source)


def _mirror_augmented_samples(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible private wrapper for the default mirror rule."""
    return mirror_augmented_samples(U)


def gaussian_kernel_conditional_mean(
    U: np.ndarray,
    Y: np.ndarray,
    bandwidth: float = DEFAULT_GAUSSIAN_BANDWIDTH,
    standardize: bool = True,
    leave_one_out: bool = True,
    boundary: str = "mirror",
    mirror_max_dim: int | None = DEFAULT_MIRROR_MAX_DIM,
    radius_scale: float = DEFAULT_KERNEL_RADIUS_SCALE,
) -> np.ndarray:
    """Estimate E[Y|U] with a fixed-bandwidth Gaussian kernel smoother."""
    return _gaussian_kernel_predict(
        U_train=U,
        Y_train=Y,
        U_query=U,
        bandwidth=bandwidth,
        standardize=standardize,
        leave_one_out=leave_one_out,
        boundary=boundary,
        mirror_max_dim=mirror_max_dim,
        radius_scale=radius_scale,
    )


def _gaussian_kernel_predict(
    U_train: np.ndarray,
    Y_train: np.ndarray,
    U_query: np.ndarray,
    bandwidth: float = DEFAULT_GAUSSIAN_BANDWIDTH,
    standardize: bool = True,
    leave_one_out: bool = False,
    boundary: str = "mirror",
    mirror_max_dim: int | None = DEFAULT_MIRROR_MAX_DIM,
    radius_scale: float = DEFAULT_KERNEL_RADIUS_SCALE,
) -> np.ndarray:
    """Predict a fixed-bandwidth Gaussian kernel conditional mean at query points."""
    U_train = _as_2d_u(U_train)
    U_query = _as_2d_u(U_query)
    Y_train = np.asarray(Y_train, dtype=float).reshape(-1)
    n_train = U_train.shape[0]
    n_query = U_query.shape[0]
    if Y_train.shape[0] != n_train:
        raise ValueError("U_train and Y_train must have the same number of samples.")
    if U_query.shape[1] != U_train.shape[1]:
        raise ValueError("U_train and U_query must have the same number of columns.")
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive.")
    if radius_scale <= 0.0:
        raise ValueError("radius_scale must be positive.")
    if leave_one_out and n_query != n_train:
        raise ValueError("leave_one_out=True requires U_query to match U_train.")
    if boundary not in {"none", "mirror"}:
        raise ValueError("boundary must be 'none' or 'mirror'.")

    if boundary == "mirror":
        U_fit, source_idx = mirror_augmented_samples(U_train, max_dim=mirror_max_dim)
    else:
        U_fit = U_train
        source_idx = np.arange(n_train)

    if standardize:
        scaler = StandardScaler().fit(U_train)
        U_fit_work = scaler.transform(U_fit)
        U_query_work = scaler.transform(U_query)
    else:
        U_fit_work = U_fit
        U_query_work = U_query

    radius = float(radius_scale * bandwidth)
    nn = NearestNeighbors(radius=radius, algorithm="auto")
    nn.fit(U_fit_work)
    distances, indices = nn.radius_neighbors(U_query_work, return_distance=True)

    fallback_neighbors = min(max(8, 2 * U_train.shape[1] + 2), U_fit_work.shape[0])
    fallback_nn = NearestNeighbors(n_neighbors=fallback_neighbors, algorithm="auto")
    fallback_nn.fit(U_fit_work)
    fallback_distances, fallback_indices = fallback_nn.kneighbors(
        U_query_work,
        return_distance=True,
    )

    m_hat = np.empty(n_query, dtype=float)
    eps = np.finfo(float).eps
    for i in range(n_query):
        row_idx = indices[i]
        row_dist = distances[i]
        if leave_one_out:
            keep = source_idx[row_idx] != i
            row_idx = row_idx[keep]
            row_dist = row_dist[keep]
        if row_idx.shape[0] == 0:
            row_idx = fallback_indices[i]
            row_dist = fallback_distances[i]
            if leave_one_out:
                keep = source_idx[row_idx] != i
                row_idx = row_idx[keep]
                row_dist = row_dist[keep]
            if row_idx.shape[0] == 0:
                raise RuntimeError("No valid Gaussian kernel neighbors are available.")

        weights = np.exp(-0.5 * (row_dist / max(float(bandwidth), eps)) ** 2)
        weight_sum = float(np.sum(weights))
        if weight_sum <= eps:
            m_hat[i] = np.mean(Y_train[source_idx[row_idx]])
        else:
            m_hat[i] = np.sum(weights * Y_train[source_idx[row_idx]]) / weight_sum
    return m_hat


def _validate_lowess_inputs(
    U_train: np.ndarray,
    Y_train: np.ndarray,
    U_query: np.ndarray,
    frac: float,
    it: int,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U_train = _as_2d_u(U_train)
    U_query = _as_2d_u(U_query)
    Y_train = np.asarray(Y_train, dtype=float).reshape(-1)
    if Y_train.shape[0] != U_train.shape[0]:
        raise ValueError("U_train and Y_train must have the same number of samples.")
    if U_train.shape[1] != 1 or U_query.shape[1] != 1:
        raise ValueError("LOWESS conditional mean requires one-dimensional U.")
    if not (0.0 < frac <= 1.0):
        raise ValueError("LOWESS frac must be in (0, 1].")
    if it < 0:
        raise ValueError("LOWESS it must be nonnegative.")
    if delta < 0.0:
        raise ValueError("LOWESS delta must be nonnegative.")
    return U_train.reshape(-1), Y_train, U_query.reshape(-1)


def _lowess_predict(
    U_train: np.ndarray,
    Y_train: np.ndarray,
    U_query: np.ndarray,
    frac: float = DEFAULT_LOWESS_FRAC,
    it: int = DEFAULT_LOWESS_IT,
    delta: float = DEFAULT_LOWESS_DELTA,
) -> np.ndarray:
    """Predict a one-dimensional LOWESS conditional mean at query points."""
    if lowess is None:
        raise ImportError("LOWESS estimator requires statsmodels. Install SI_DL-main/requirements.txt.")
    x_train, y_train, x_query = _validate_lowess_inputs(
        U_train,
        Y_train,
        U_query,
        frac=frac,
        it=it,
        delta=delta,
    )
    return np.asarray(
        lowess(
            endog=y_train,
            exog=x_train,
            frac=float(frac),
            it=int(it),
            delta=float(delta),
            xvals=x_query,
            is_sorted=False,
            return_sorted=False,
        ),
        dtype=float,
    ).reshape(-1)


def lowess_conditional_mean(
    U: np.ndarray,
    Y: np.ndarray,
    frac: float = DEFAULT_LOWESS_FRAC,
    it: int = DEFAULT_LOWESS_IT,
    delta: float = DEFAULT_LOWESS_DELTA,
    leave_one_out: bool = True,
) -> np.ndarray:
    """Estimate E[Y|U] with statsmodels LOWESS for one-dimensional U."""
    x, y, _ = _validate_lowess_inputs(
        U,
        Y,
        U,
        frac=frac,
        it=it,
        delta=delta,
    )
    if not leave_one_out:
        return _lowess_predict(x.reshape(-1, 1), y, x.reshape(-1, 1), frac=frac, it=it, delta=delta)

    n = x.shape[0]
    if n < 3:
        raise ValueError("At least three samples are required for leave-one-out LOWESS.")
    m_hat = np.empty(n, dtype=float)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        m_hat[i] = _lowess_predict(
            x[keep].reshape(-1, 1),
            y[keep],
            np.array([[x[i]]], dtype=float),
            frac=frac,
            it=it,
            delta=delta,
        )[0]
    return m_hat


def stable_scov(Y: np.ndarray, m_hat: np.ndarray) -> dict:
    """Compute the covariance SI-DL score with centered dot products."""
    Y = np.asarray(Y, dtype=float).reshape(-1)
    m_hat = np.asarray(m_hat, dtype=float).reshape(-1)
    if Y.shape[0] != m_hat.shape[0]:
        raise ValueError("Y and m_hat must have the same number of samples.")
    if Y.shape[0] < 2:
        raise ValueError("At least two samples are required.")

    y_centered = Y - np.mean(Y)
    m_centered = m_hat - np.mean(m_hat)
    var_y_sum = float(np.dot(y_centered, y_centered))
    if var_y_sum <= np.finfo(float).eps:
        raise ValueError("Var(Y) must be positive.")

    cov_sum = float(np.dot(y_centered, m_centered))
    scov_raw = cov_sum / var_y_sum
    scov = float(np.clip(scov_raw, 0.0, 1.0))
    return {
        "S_cov": scov,
        "S_cov_raw": scov_raw,
        "Var_Y": var_y_sum / (Y.shape[0] - 1),
        "Cov_Y_mhat": cov_sum / (Y.shape[0] - 1),
    }


def explained_variance_score(
    U: np.ndarray,
    Y: np.ndarray,
    bandwidth: float = DEFAULT_GAUSSIAN_BANDWIDTH,
    estimator: str = "gaussian_kernel",
    standardize: bool = True,
    leave_one_out: bool = True,
    boundary: str = "mirror",
    mirror_max_dim: int | None = DEFAULT_MIRROR_MAX_DIM,
    radius_scale: float = DEFAULT_KERNEL_RADIUS_SCALE,
    lowess_frac: float = DEFAULT_LOWESS_FRAC,
    lowess_it: int = DEFAULT_LOWESS_IT,
    lowess_delta: float = DEFAULT_LOWESS_DELTA,
) -> dict:
    """
    Estimate S(U) = Var(E[Y|U]) / Var(Y) with a nonparametric mean estimator.

    The returned S_cov uses the stable covariance identity
    Cov(Y, m_hat(U)) / Var(Y). The default estimator is the leave-one-out
    Gaussian kernel conditional mean. Set estimator="lowess" to use
    statsmodels.nonparametric.smoothers_lowess.lowess for one-dimensional U.
    """
    U = _as_2d_u(U)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    if Y.shape[0] != U.shape[0]:
        raise ValueError("U and Y must have the same number of samples.")

    if estimator == "gaussian_kernel":
        m_hat = gaussian_kernel_conditional_mean(
            U,
            Y,
            standardize=standardize,
            bandwidth=bandwidth,
            leave_one_out=leave_one_out,
            boundary=boundary,
            mirror_max_dim=mirror_max_dim,
            radius_scale=radius_scale,
        )
        estimator_details = {
            "bandwidth": float(bandwidth),
            "standardize": bool(standardize),
            "radius_scale": float(radius_scale),
            "boundary": boundary,
            "mirror_max_dim": mirror_max_dim,
        }
    elif estimator == "lowess":
        m_hat = lowess_conditional_mean(
            U,
            Y,
            frac=lowess_frac,
            it=lowess_it,
            delta=lowess_delta,
            leave_one_out=leave_one_out,
        )
        estimator_details = {
            "lowess_frac": float(lowess_frac),
            "lowess_it": int(lowess_it),
            "lowess_delta": float(lowess_delta),
        }
    else:
        raise ValueError("estimator must be 'gaussian_kernel' or 'lowess'.")

    score = stable_scov(Y, m_hat)
    score.update({
        "m_hat": m_hat,
        "estimator": estimator,
        "leave_one_out": bool(leave_one_out),
        "n_retained": int(Y.shape[0]),
    })
    score.update(estimator_details)
    return score


def cross_validate_bandwidth(
    U: np.ndarray,
    Y: np.ndarray,
    bandwidths: np.ndarray | list[float] | None = None,
    cv: int = 5,
    random_state: int | None = None,
    standardize: bool = True,
    boundary: str = "mirror",
    mirror_max_dim: int | None = DEFAULT_MIRROR_MAX_DIM,
    radius_scale: float = DEFAULT_KERNEL_RADIUS_SCALE,
) -> dict:
    """Select Gaussian bandwidth by K-fold prediction-error cross-validation."""
    U = _as_2d_u(U)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    if Y.shape[0] != U.shape[0]:
        raise ValueError("U and Y must have the same number of samples.")
    if cv < 2:
        raise ValueError("cv must be at least 2.")

    if bandwidths is None:
        bandwidths = DEFAULT_BANDWIDTH_GRID
    bandwidths = np.asarray(bandwidths, dtype=float).reshape(-1)
    if bandwidths.size == 0 or np.any(bandwidths <= 0.0):
        raise ValueError("bandwidths must contain positive values.")

    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows = []
    for bandwidth in bandwidths:
        fold_mse = []
        for train_idx, valid_idx in splitter.split(U):
            pred = _gaussian_kernel_predict(
                U_train=U[train_idx],
                Y_train=Y[train_idx],
                U_query=U[valid_idx],
                bandwidth=float(bandwidth),
                standardize=standardize,
                leave_one_out=False,
                boundary=boundary,
                mirror_max_dim=mirror_max_dim,
                radius_scale=radius_scale,
            )
            fold_mse.append(float(np.mean((Y[valid_idx] - pred) ** 2)))
        rows.append({
            "bandwidth": float(bandwidth),
            "cv_mse": float(np.mean(fold_mse)),
            "cv_mse_std": float(np.std(fold_mse, ddof=1)) if len(fold_mse) > 1 else 0.0,
        })

    best = min(rows, key=lambda row: (row["cv_mse"], row["bandwidth"]))
    return {
        "best_bandwidth": best["bandwidth"],
        "best_cv_mse": best["cv_mse"],
        "bandwidths": bandwidths,
        "cv_results": rows,
        "cv": int(cv),
    }


def cross_validate_lowess_frac(
    U: np.ndarray,
    Y: np.ndarray,
    fracs: np.ndarray | list[float] | None = None,
    cv: int = 5,
    random_state: int | None = None,
    lowess_it: int = DEFAULT_LOWESS_IT,
    lowess_delta: float = DEFAULT_LOWESS_DELTA,
) -> dict:
    """Select LOWESS frac by K-fold prediction-error cross-validation."""
    U = _as_2d_u(U)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    if Y.shape[0] != U.shape[0]:
        raise ValueError("U and Y must have the same number of samples.")
    if U.shape[1] != 1:
        raise ValueError("LOWESS frac CV requires one-dimensional U.")
    if cv < 2:
        raise ValueError("cv must be at least 2.")

    if fracs is None:
        fracs = DEFAULT_LOWESS_FRAC_GRID
    fracs = np.asarray(fracs, dtype=float).reshape(-1)
    if fracs.size == 0 or np.any(fracs <= 0.0) or np.any(fracs > 1.0):
        raise ValueError("fracs must contain values in (0, 1].")

    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows = []
    for frac in fracs:
        fold_mse = []
        for train_idx, valid_idx in splitter.split(U):
            pred = _lowess_predict(
                U_train=U[train_idx],
                Y_train=Y[train_idx],
                U_query=U[valid_idx],
                frac=float(frac),
                it=lowess_it,
                delta=lowess_delta,
            )
            fold_mse.append(float(np.mean((Y[valid_idx] - pred) ** 2)))
        rows.append({
            "lowess_frac": float(frac),
            "cv_mse": float(np.mean(fold_mse)),
            "cv_mse_std": float(np.std(fold_mse, ddof=1)) if len(fold_mse) > 1 else 0.0,
        })

    best = min(rows, key=lambda row: (row["cv_mse"], row["lowess_frac"]))
    return {
        "best_lowess_frac": best["lowess_frac"],
        "best_cv_mse": best["cv_mse"],
        "lowess_fracs": fracs,
        "cv_results": rows,
        "cv": int(cv),
    }


def cross_validated_explained_variance_score(
    U: np.ndarray,
    Y: np.ndarray,
    bandwidths: np.ndarray | list[float] | None = None,
    estimator: str = "gaussian_kernel",
    cv: int = 5,
    random_state: int | None = None,
    standardize: bool = True,
    boundary: str = "mirror",
    mirror_max_dim: int | None = DEFAULT_MIRROR_MAX_DIM,
    radius_scale: float = DEFAULT_KERNEL_RADIUS_SCALE,
    lowess_fracs: np.ndarray | list[float] | None = None,
    lowess_it: int = DEFAULT_LOWESS_IT,
    lowess_delta: float = DEFAULT_LOWESS_DELTA,
) -> dict:
    """Select smoothing hyperparameter by CV, then compute the LOO S_cov score."""
    if estimator == "lowess":
        cv_result = cross_validate_lowess_frac(
            U,
            Y,
            fracs=lowess_fracs,
            cv=cv,
            random_state=random_state,
            lowess_it=lowess_it,
            lowess_delta=lowess_delta,
        )
        score = explained_variance_score(
            U,
            Y,
            estimator="lowess",
            leave_one_out=True,
            lowess_frac=cv_result["best_lowess_frac"],
            lowess_it=lowess_it,
            lowess_delta=lowess_delta,
        )
        score.update({
            "lowess_frac_cv": True,
            "lowess_frac_grid": np.asarray(cv_result["lowess_fracs"], dtype=float),
            "lowess_frac_cv_results": cv_result["cv_results"],
            "lowess_frac_cv_folds": cv_result["cv"],
            "lowess_frac_cv_mse": cv_result["best_cv_mse"],
        })
        return score
    if estimator != "gaussian_kernel":
        raise ValueError("estimator must be 'gaussian_kernel' or 'lowess'.")

    cv_result = cross_validate_bandwidth(
        U,
        Y,
        bandwidths=bandwidths,
        cv=cv,
        random_state=random_state,
        standardize=standardize,
        boundary=boundary,
        mirror_max_dim=mirror_max_dim,
        radius_scale=radius_scale,
    )
    score = explained_variance_score(
        U,
        Y,
        bandwidth=cv_result["best_bandwidth"],
        estimator="gaussian_kernel",
        standardize=standardize,
        leave_one_out=True,
        boundary=boundary,
        mirror_max_dim=mirror_max_dim,
        radius_scale=radius_scale,
    )
    score.update({
        "bandwidth_cv": True,
        "bandwidth_grid": np.asarray(cv_result["bandwidths"], dtype=float),
        "bandwidth_cv_results": cv_result["cv_results"],
        "bandwidth_cv_folds": cv_result["cv"],
        "bandwidth_cv_mse": cv_result["best_cv_mse"],
    })
    return score


def format_score_estimator(score: dict) -> str:
    """Format the active SI-DL score estimator for display."""
    if score.get("estimator") == "lowess":
        return f"lowess, frac={score.get('lowess_frac', DEFAULT_LOWESS_FRAC):.4g}"
    if "bandwidth" in score:
        return f"{score.get('estimator', 'gaussian_kernel')}, bandwidth={score['bandwidth']:.4g}"
    return score.get("estimator", "gaussian_kernel")


def _normalize_omega(omega: np.ndarray) -> np.ndarray:
    omega = np.asarray(omega, dtype=float).reshape(-1)
    max_abs = float(np.max(np.abs(omega)))
    if max_abs <= 1e-12:
        return omega
    omega = omega / max_abs
    first = np.flatnonzero(np.abs(omega) > 1e-10)
    if first.size and omega[first[0]] < 0.0:
        omega = -omega
    return omega


def params_to_omegas(params: np.ndarray, basis_matrices: np.ndarray, num_input: int) -> np.ndarray:
    """Map differential-evolution parameters to normalized dimensionless exponents."""
    params = np.asarray(params, dtype=float).reshape(-1)
    basis_matrices = np.asarray(basis_matrices, dtype=float)
    num_basis = basis_matrices.shape[0]
    omegas = []
    for i in range(num_input):
        a = params[i * num_basis : (i + 1) * num_basis]
        omega = a @ basis_matrices
        omegas.append(_normalize_omega(omega))
    return np.asarray(omegas)


def optimize_dimensionless_groups(
    X: np.ndarray,
    Y: np.ndarray,
    basis_matrices: np.ndarray,
    num_input: int = 1,
    bandwidth: float = DEFAULT_GAUSSIAN_BANDWIDTH,
    bandwidths: np.ndarray | list[float] | None = None,
    estimator: str = "gaussian_kernel",
    cv: int = 5,
    bounds: tuple[float, float] = (-2.0, 2.0),
    maxiter: int = 60,
    popsize: int = 10,
    seed: int | None = None,
    search_size: int | None = None,
    polish: bool = True,
    lowess_frac: float = DEFAULT_LOWESS_FRAC,
    lowess_fracs: np.ndarray | list[float] | None = None,
    lowess_it: int = DEFAULT_LOWESS_IT,
    lowess_delta: float = DEFAULT_LOWESS_DELTA,
) -> dict:
    """Find dimensionless Pi groups that maximize the SI-DL S_cov score."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    basis_matrices = np.asarray(basis_matrices, dtype=float)
    if np.any(X <= 0.0):
        raise ValueError("SI-DL dimensionless search expects positive dimensional inputs.")
    rng = np.random.default_rng(seed)
    if search_size is not None and search_size < X.shape[0]:
        search_idx = rng.choice(X.shape[0], size=search_size, replace=False)
        X_search = X[search_idx]
        Y_search = Y[search_idx]
    else:
        search_idx = None
        X_search = X
        Y_search = Y

    num_basis = basis_matrices.shape[0]
    dim = num_basis * num_input
    de_bounds = [bounds] * dim

    def objective(params: np.ndarray) -> float:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            omegas = params_to_omegas(params, basis_matrices, num_input)
            if np.any(np.max(np.abs(omegas), axis=1) <= 1e-12):
                return 1e6
            U = calc_log_pi_omega(omegas, X_search)
            try:
                if estimator == "lowess":
                    if lowess_fracs is None:
                        score = explained_variance_score(
                            U,
                            Y_search,
                            estimator="lowess",
                            lowess_frac=lowess_frac,
                            lowess_it=lowess_it,
                            lowess_delta=lowess_delta,
                        )["S_cov"]
                    else:
                        score = cross_validated_explained_variance_score(
                            U,
                            Y_search,
                            estimator="lowess",
                            cv=cv,
                            random_state=seed,
                            lowess_fracs=lowess_fracs,
                            lowess_it=lowess_it,
                            lowess_delta=lowess_delta,
                        )["S_cov"]
                elif estimator == "gaussian_kernel":
                    if bandwidths is None:
                        score = explained_variance_score(
                            U,
                            Y_search,
                            bandwidth=bandwidth,
                        )["S_cov"]
                    else:
                        score = cross_validated_explained_variance_score(
                            U,
                            Y_search,
                            bandwidths=bandwidths,
                            cv=cv,
                            random_state=seed,
                        )["S_cov"]
                else:
                    raise ValueError("estimator must be 'gaussian_kernel' or 'lowess'.")
            except Exception:
                return 1e6
            if not np.isfinite(score):
                return 1e6
            return -float(score)

    result = differential_evolution(
        objective,
        bounds=de_bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=polish,
        updating="immediate",
        workers=1,
    )

    omegas = params_to_omegas(result.x, basis_matrices, num_input)
    input_log_PI = calc_log_pi_omega(omegas, X)
    input_PI = np.exp(input_log_PI)
    if estimator == "lowess":
        if lowess_fracs is None:
            score_result = explained_variance_score(
                input_log_PI,
                Y,
                estimator="lowess",
                lowess_frac=lowess_frac,
                lowess_it=lowess_it,
                lowess_delta=lowess_delta,
            )
            individual_scores = [
                explained_variance_score(
                    input_log_PI[:, j],
                    Y,
                    estimator="lowess",
                    lowess_frac=lowess_frac,
                    lowess_it=lowess_it,
                    lowess_delta=lowess_delta,
                )
                for j in range(num_input)
            ]
        else:
            score_result = cross_validated_explained_variance_score(
                input_log_PI,
                Y,
                estimator="lowess",
                cv=cv,
                random_state=seed,
                lowess_fracs=lowess_fracs,
                lowess_it=lowess_it,
                lowess_delta=lowess_delta,
            )
            individual_scores = [
                cross_validated_explained_variance_score(
                    input_log_PI[:, j],
                    Y,
                    estimator="lowess",
                    cv=cv,
                    random_state=seed,
                    lowess_fracs=lowess_fracs,
                    lowess_it=lowess_it,
                    lowess_delta=lowess_delta,
                )
                for j in range(num_input)
            ]
    elif estimator == "gaussian_kernel":
        if bandwidths is None:
            score_result = explained_variance_score(
                input_log_PI,
                Y,
                bandwidth=bandwidth,
            )
            individual_scores = [
                explained_variance_score(
                    input_log_PI[:, j],
                    Y,
                    bandwidth=bandwidth,
                )
                for j in range(num_input)
            ]
        else:
            score_result = cross_validated_explained_variance_score(
                input_log_PI,
                Y,
                bandwidths=bandwidths,
                cv=cv,
                random_state=seed,
            )
            individual_scores = [
                cross_validated_explained_variance_score(
                    input_log_PI[:, j],
                    Y,
                    bandwidths=bandwidths,
                    cv=cv,
                    random_state=seed,
                )
                for j in range(num_input)
            ]
    else:
        raise ValueError("estimator must be 'gaussian_kernel' or 'lowess'.")

    return {
        "input_coef": omegas,
        "input_coef_basis": result.x.reshape(num_input, num_basis),
        "input_log_PI": input_log_PI,
        "input_PI": input_PI,
        "output_PI": Y.reshape(-1, 1),
        "score": score_result,
        "individual_scores": individual_scores,
        "optimizer_result": result,
        "search_idx": search_idx,
    }


def main(
    X: np.ndarray,
    Y: np.ndarray,
    basis_matrices: np.ndarray,
    num_input: int = 1,
    bandwidth: float = DEFAULT_GAUSSIAN_BANDWIDTH,
    bandwidths: np.ndarray | list[float] | None = None,
    estimator: str = "gaussian_kernel",
    cv: int = 5,
    bounds: tuple[float, float] = (-2.0, 2.0),
    maxiter: int = 60,
    popsize: int = 10,
    seed: int | None = None,
    search_size: int | None = None,
    polish: bool = True,
    lowess_frac: float = DEFAULT_LOWESS_FRAC,
    lowess_fracs: np.ndarray | list[float] | None = None,
    lowess_it: int = DEFAULT_LOWESS_IT,
    lowess_delta: float = DEFAULT_LOWESS_DELTA,
) -> dict:
    """IT_PI-style entry point, using SI-DL instead of mutual information."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    basis_matrices = np.asarray(basis_matrices, dtype=float)
    print("-" * 60)
    print("SI-DL dimensionless search")
    print("num of parameters:", basis_matrices.shape[0] * num_input)
    if estimator == "lowess":
        if lowess_fracs is None:
            estimator_text = f"frac={lowess_frac:.4g}"
        else:
            fracs_arr = np.asarray(lowess_fracs, dtype=float)
            estimator_text = (
                f"frac CV over [{fracs_arr.min():.4g}, {fracs_arr.max():.4g}], "
                f"cv={cv}"
            )
    elif bandwidths is None:
        estimator_text = f"bandwidth={bandwidth:.4g}"
    else:
        bandwidths_arr = np.asarray(bandwidths, dtype=float)
        estimator_text = (
            f"bandwidth CV over [{bandwidths_arr.min():.4g}, {bandwidths_arr.max():.4g}], "
            f"cv={cv}"
        )
    print(
        f"estimator={estimator}, {estimator_text}, "
        f"optimizer=differential_evolution, maxiter={maxiter}, popsize={popsize}"
    )
    if search_size is not None:
        print(f"optimization subset size: {min(search_size, np.asarray(X).shape[0])}")

    results = optimize_dimensionless_groups(
        X=X,
        Y=Y,
        basis_matrices=basis_matrices,
        num_input=num_input,
        bandwidth=bandwidth,
        bandwidths=bandwidths,
        estimator=estimator,
        cv=cv,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        search_size=search_size,
        polish=polish,
        lowess_frac=lowess_frac,
        lowess_fracs=lowess_fracs,
        lowess_it=lowess_it,
        lowess_delta=lowess_delta,
    )
    print("optimized S_cov:", results["score"]["S_cov"])
    print("score estimator:", format_score_estimator(results["score"]))
    print("optimized exponent vectors:")
    for row in results["input_coef"]:
        print(np.round(row, 4))
    print("-" * 60)
    return results


def savefig(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_1d_fit(U: np.ndarray, Y: np.ndarray, result: dict, path: str | Path, title: str) -> None:
    U = np.asarray(U, dtype=float).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)

    plt.figure(figsize=(6.2, 4.4))
    plt.scatter(U, Y, s=14, alpha=0.55, color="tab:blue")
    plt.xlabel(r"$\pi_1$", fontsize=PLOT_LABEL_FONTSIZE)
    plt.ylabel(r"$\pi_0$", fontsize=PLOT_LABEL_FONTSIZE)
    plt.title(title, fontsize=PLOT_TITLE_FONTSIZE)
    plt.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)
    plt.grid(True, alpha=0.3)
    savefig(path)


def plot_2d_fit(U: np.ndarray, Y: np.ndarray, m_hat: np.ndarray, path: str | Path, title: str) -> None:
    U = np.asarray(U, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)

    rng = np.random.default_rng(20260601)
    n_plot = min(3500, U.shape[0])
    idx = rng.choice(U.shape[0], size=n_plot, replace=False)
    vmin = float(np.min(Y))
    vmax = float(np.max(Y))

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    scatter = ax.scatter(
        U[idx, 0],
        U[idx, 1],
        c=Y[idx],
        s=12,
        alpha=0.75,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(r"$\pi_1$", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\pi_2$", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=PLOT_TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.85)
    cbar.set_label(r"$\pi_0$", fontsize=PLOT_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=PLOT_COLORBAR_FONTSIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_2d_result_3d(U: np.ndarray, Y: np.ndarray, path: str | Path, title: str) -> None:
    U = np.asarray(U, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)

    rng = np.random.default_rng(20260601)
    n_plot = min(3500, U.shape[0])
    idx = rng.choice(U.shape[0], size=n_plot, replace=False)

    fig = plt.figure(figsize=(8.0, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        U[idx, 0],
        U[idx, 1],
        Y[idx],
        c=Y[idx],
        s=10,
        alpha=0.72,
        cmap="viridis",
        depthshade=False,
    )
    ax.set_xlabel(r"$\pi_1$", labelpad=10, fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\pi_2$", labelpad=10, fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_zlabel(r"$\pi_0$", labelpad=10, fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_title(title, pad=16, fontsize=PLOT_TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)
    ax.view_init(elev=24, azim=-128)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.70, pad=0.10)
    cbar.set_label(r"$\pi_0$", fontsize=PLOT_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=PLOT_COLORBAR_FONTSIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_score_bars(labels: list[str], scores: list[float], path: str | Path, title: str) -> None:
    plt.figure(figsize=(7.4, 4.4))
    bars = plt.bar(labels, scores, color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"][: len(labels)])
    plt.ylabel("S_cov")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.ylim(min(0.0, min(scores) - 0.05), max(1.05, max(scores) + 0.05))
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{score:.3f}", ha="center", va="bottom")
    savefig(path)
