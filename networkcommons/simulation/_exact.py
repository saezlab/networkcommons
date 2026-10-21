"""
Structural equilibria and attracting stationary laws of affine mechanisms.
"""

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from ._model import (
    CyclicSystem, checked, free_indices, frozen_array, integer,
    production, project, scalar, vector,
)


@dataclass(frozen = True, eq = False)
class EquilibriumResult:
    """Structural equilibrium and checked free and clamp residuals."""

    state: np.ndarray
    free_residual: np.ndarray
    clamp_residual: np.ndarray
    iterations: int
    metadata: Mapping[str, Any]


@dataclass(frozen = True, eq = False)
class StationaryResult:
    """Independent Gaussian stationary samples and their generating moments."""

    samples: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    metadata: Mapping[str, Any]


def affine_system(system):
    """Return reduced K and input, without evaluating discarded mechanisms."""
    free = free_indices(system)
    n = len(system.basal)
    matrix = np.zeros((len(free), n))
    d = np.empty(len(free))
    with np.errstate(over = 'ignore', invalid = 'ignore'):
        for row, i in enumerate(free):
            p = system.production_scale[i]
            if p == 0:
                d[row] = system.production_offset[i]
                continue
            if not system.transfer[i].is_identity:
                raise ValueError(f'Free receiving mechanism {i} is not affine')
            matrix[row] = p * system.interaction_matrix[i]
            d[row] = system.production_offset[i] + p * system.basal[i]
        for i, value in system.clamps.items():
            d += matrix[:, i] * value
        k = np.eye(len(free)) - matrix[:, free]
    checked(k, 'reduced affine operator')
    checked(d, 'reduced affine input')
    return free, k, d


def residuals(system, state):
    free = free_indices(system)
    values = production(system, state)
    residual = np.zeros_like(state)
    with np.errstate(over = 'ignore', invalid = 'ignore'):
        residual[free] = state[free] - values[free]
    checked(residual, 'equilibrium residual')
    clamp_residual = np.zeros_like(state)
    for i, value in system.clamps.items():
        clamp_residual[i] = state[i] - value
    return values, residual, clamp_residual


def solve_equilibrium(
    system: CyclicSystem, *,
    method: Literal['fixed_point', 'linear'] = 'fixed_point',
    initial_state: ArrayLike | None = None,
    relaxation: float | None = None, max_iterations: int | None = None,
    atol: float = 1e-10, rtol: float = 1e-8,
) -> EquilibriumResult:
    """
    Solve structural equations, with explicit residual-based acceptance.

    Args:
        system: Validated CyclicSystem.
        method: 'fixed_point' or a direct affine 'linear' solve.
        initial_state: Required length-n vector for fixed-point free systems.
        relaxation: Fixed-point damping in (0, 1], default 0.5.
        max_iterations: Positive fixed-point update limit, default 10000.
        atol: Positive absolute free-equation tolerance.
        rtol: Nonnegative relative free-equation tolerance.

    Returns:
        EquilibriumResult. This is not a nonlinear stochastic population mean.

    Raises:
        ValueError: Invalid arguments or non-affine linear solve.
        RuntimeError: Fixed-point residual tolerance is not reached.
        numpy.linalg.LinAlgError: Singular affine equations.
    """
    if method not in ('fixed_point', 'linear'):
        raise ValueError('method must be fixed_point or linear')
    atol = scalar(atol, 'atol', positive = True)
    rtol = scalar(rtol, 'rtol', minimum = 0)
    free = free_indices(system)
    n = len(system.basal)
    initial = (
        None if initial_state is None
        else vector(initial_state, 'initial_state', n)
    )
    if method == 'linear':
        unused = (initial_state, relaxation, max_iterations)
        if any(v is not None for v in unused):
            raise ValueError(
                'linear method does not use initial_state, relaxation, '
                'or max_iterations',
            )
        damping = limit = None
    else:
        damping = (
            0.5 if relaxation is None else
            scalar(relaxation, 'relaxation', positive = True)
        )
        if damping > 1:
            raise ValueError('relaxation must be <= 1')
        limit = (
            10000 if max_iterations is None else
            integer(max_iterations, 'max_iterations', minimum = 1)
        )
        if len(free) and initial is None:
            raise ValueError(
                'fixed_point requires initial_state for free coordinates',
            )
    state = np.zeros(n) if initial is None else initial.copy()
    project(system, state)
    iterations = 0
    if len(free) and method == 'linear':
        _, k, d = affine_system(system)
        state[free] = np.linalg.solve(k, d)
        checked(state, 'linear equilibrium')
    while True:
        values, residual, clamp_residual = residuals(system, state)
        with np.errstate(over = 'ignore', invalid = 'ignore'):
            threshold = atol + rtol * np.maximum(np.abs(state), np.abs(values))
        if np.all(np.abs(residual[free]) <= threshold[free]):
            break
        if method == 'linear':
            raise RuntimeError(
                'Linear equilibrium failed the requested residual tolerance',
            )
        if iterations >= limit:
            raise RuntimeError(
                f'Equilibrium did not converge after {iterations} iterations. '
                f'Maximum free residual: {np.max(np.abs(residual[free]))}',
            )
        with np.errstate(over = 'ignore', invalid = 'ignore'):
            state[free] = (
                (1 - damping) * state[free] + damping * values[free]
            )
        iterations += 1
        project(system, state)
        checked(state, 'fixed-point state', step = iterations)
    return EquilibriumResult(
        state = frozen_array(state), free_residual = frozen_array(residual),
        clamp_residual = frozen_array(clamp_residual), iterations = iterations,
        metadata = MappingProxyType({
            'method': method, 'atol': atol, 'rtol': rtol,
            'relaxation': damping, 'max_iterations': limit,
        }),
    )


def sample_linear_stationary(
    system: CyclicSystem, *, n_samples: int, process_seed: int,
    atol: float = 1e-12, rtol: float = 1e-10,
) -> StationaryResult:
    """
    Sample the attracting Gaussian law of the reduced affine dynamics.

    Args:
        system: Model with identity transfers on active free receivers.
        n_samples: Positive integer number of independent samples.
        process_seed: Nonnegative integer for a local PCG64 generator.
        atol: Positive absolute covariance and residual tolerance.
        rtol: Nonnegative relative covariance and residual tolerance.

    Returns:
        StationaryResult with analytic mean, corrected PSD covariance, and
        numerical stability and Lyapunov diagnostics. No jitter is added.

    Raises:
        ValueError: Non-affine mechanisms or non-attracting free dynamics.
        RuntimeError: Covariance positivity or residual checks fail.
    """
    from scipy.linalg import norm, solve_continuous_lyapunov

    count = integer(n_samples, 'n_samples', minimum = 1)
    seed = integer(process_seed, 'process_seed')
    atol = scalar(atol, 'atol', positive = True)
    rtol = scalar(rtol, 'rtol', minimum = 0)
    free, k, d = affine_system(system)
    n = len(system.basal)
    mean, covariance = np.zeros(n), np.zeros((n, n))
    project(system, mean)
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.tile(mean, (count, 1))
    margin, residual, clipped = None, 0.0, 0.0
    if len(free):
        eigenvalues = np.linalg.eigvals(k)
        checked(eigenvalues, 'operator eigenvalues')
        margin = float(np.min(eigenvalues.real))
        if margin <= 0:
            raise ValueError(
                'Stationary sampling requires positive real eigenvalues '
                'of reduced K',
            )
        mean[free] = np.linalg.solve(k, d)
        with np.errstate(over = 'ignore', invalid = 'ignore'):
            forcing = np.diag(system.diffusion[free] ** 2)
        checked(forcing, 'diffusion covariance')
        cov = solve_continuous_lyapunov(k, forcing)
        checked(cov, 'Lyapunov covariance')
        cov = cov / 2 + cov.T / 2
        vals, vectors = np.linalg.eigh(cov)
        allowance = atol + rtol * np.linalg.norm(cov, ord = 2)
        if np.min(vals) < -allowance:
            raise RuntimeError(
                'Stationary covariance is not positive semidefinite',
            )
        clipped = float(max(0, -np.min(vals)))
        factor = vectors * np.sqrt(np.maximum(vals, 0))
        with np.errstate(over = 'ignore', invalid = 'ignore'):
            cov = factor @ factor.T
        checked(cov, 'reconstructed covariance')
        with np.errstate(over = 'ignore', invalid = 'ignore'):
            defect = k @ cov + cov @ k.T - forcing
        checked(defect, 'Lyapunov residual')
        # A scaled Euclidean norm avoids overflow from squaring finite entries.
        residual = float(norm(defect.ravel()))
        forcing_norm = float(norm(forcing.ravel()))
        checked(np.array([residual, forcing_norm]), 'Lyapunov norms')
        if residual > atol + rtol * forcing_norm:
            raise RuntimeError(
                'Stationary covariance failed the Lyapunov residual tolerance',
            )
        covariance[np.ix_(free, free)] = cov
        with np.errstate(over = 'ignore', invalid = 'ignore'):
            draws = rng.standard_normal((count, len(free)))
            samples[:, free] = mean[free] + draws @ factor.T
    checked(mean, 'stationary mean')
    checked(samples, 'stationary samples')
    return StationaryResult(
        samples = frozen_array(samples), mean = frozen_array(mean),
        covariance = frozen_array(covariance),
        metadata = MappingProxyType({
            'method': 'linear_stationary', 'process_seed': seed,
            'stability_margin': margin, 'lyapunov_residual': residual,
            'largest_clipped_eigenvalue': clipped, 'atol': atol, 'rtol': rtol,
            'dtype': 'float64',
        }),
    )
