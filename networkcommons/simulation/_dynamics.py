"""
Finite-time Euler–Maruyama integration and independent Gaussian readouts.
"""

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._model import (
    CyclicSystem, checked, free_indices, frozen_array, integer,
    production, project, real_array, scalar, vector,
)


@dataclass(frozen = True, eq = False)
class DynamicsResult:
    """Finite-time latent states and optional saved trajectories."""

    initial_states: np.ndarray
    final_states: np.ndarray
    times: np.ndarray | None
    states: np.ndarray | None
    metadata: Mapping[str, Any]


@dataclass(frozen = True, eq = False)
class ObservationResult:
    """Readout signal, realized measurement noise, and measured values."""

    signal: np.ndarray
    measurement_noise: np.ndarray
    values: np.ndarray
    metadata: Mapping[str, Any]


def simulate_dynamics(
    system: CyclicSystem, *, initial_state: ArrayLike, dt: float,
    n_steps: int, process_seed: int, save_every: int | None = None,
) -> DynamicsResult:
    """
    Integrate finite-time paths with Euler–Maruyama and exact clamp projection.

    Args:
        system: Validated CyclicSystem.
        initial_state: Shape (n,) or (m, n). Rows explicitly define paths.
        dt: Positive time step.
        n_steps: Positive integer number of steps.
        process_seed: Nonnegative integer for a local PCG64 generator.
        save_every: Positive integer trace cadence, or None for endpoints only.

    Returns:
        DynamicsResult with shape (m, n) terminal states. Saved traces have
        shape (saved times, m, n), including time zero and the final step.

    Raises:
        ValueError: Invalid numerical arguments or array shapes.
        FloatingPointError: Nonfinite mechanism, drift, or state evaluation.

    The result makes no equilibrium or stationarity claim.
    """
    dt = scalar(dt, 'dt', positive = True)
    n_steps = integer(n_steps, 'n_steps', minimum = 1)
    process_seed = integer(process_seed, 'process_seed')
    try:
        duration = dt * n_steps
    except OverflowError as exc:
        raise ValueError('duration must be finite') from exc
    if not np.isfinite(duration):
        raise ValueError('duration must be finite')
    if save_every is not None:
        save_every = integer(save_every, 'save_every', minimum = 1)
    states = real_array(initial_state, 'initial_state')
    n = len(system.basal)
    if states.shape == (n,):
        states = states[None, :]
    if states.ndim != 2 or states.shape[1] != n or not states.shape[0]:
        raise ValueError(
            'initial_state must have shape (n,) or (m, n) with m >= 1',
        )
    project(system, states)
    initial = frozen_array(states)
    free = free_indices(system)
    amplitudes = np.zeros(n)
    amplitudes[free] = system.diffusion[free]
    stochastic = np.any(amplitudes > 0)
    rng = np.random.Generator(np.random.PCG64(process_seed))
    saved = [states.copy()] if save_every is not None else None
    times = [0.0] if saved is not None else None
    for step in range(1, n_steps + 1):
        time = (step - 1) * dt
        rates = production(system, states, step = step - 1, time = time)
        with np.errstate(over = 'ignore', invalid = 'ignore'):
            drift = rates[:, free] - states[:, free]
        checked(
            drift, f'drift on free nodes {free.tolist()}',
            step = step - 1, time = time,
        )
        with np.errstate(over = 'ignore', invalid = 'ignore'):
            states[:, free] += dt * drift
            if stochastic:
                draws = rng.standard_normal(states.shape)
                states[:, free] += (
                    np.sqrt(dt) * amplitudes[free] * draws[:, free]
                )
        project(system, states)
        checked(states, 'state', step = step, time = step * dt)
        if saved is not None and (step % save_every == 0 or step == n_steps):
            saved.append(states.copy())
            times.append(step * dt)
    return DynamicsResult(
        initial_states = initial,
        final_states = frozen_array(states),
        times = None if times is None else frozen_array(times),
        states = None if saved is None else frozen_array(saved),
        metadata = MappingProxyType({
            'method': 'euler_maruyama', 'dt': dt, 'n_steps': n_steps,
            'save_every': save_every, 'process_seed': process_seed,
            'dtype': 'float64', 'n_paths': len(states),
        }),
    )


def observe(
    latent_states: ArrayLike, *, operator: ArrayLike | None = None,
    noise_sd: float | ArrayLike = 0.0, measurement_seed: int,
) -> ObservationResult:
    """
    Apply a linear readout and independent additive Gaussian measurement noise.

    Args:
        latent_states: Finite nonempty array, with nodes on the last axis.
        operator: Shape (d, n) readout matrix, or None for identity.
        noise_sd: Nonnegative scalar or length-d measurement deviations.
        measurement_seed: Nonnegative integer for a local PCG64 generator.

    Returns:
        ObservationResult preserving leading input axes. The signal is a
        readout of the supplied latent states, not a population mean.
    """
    states = real_array(latent_states, 'latent_states')
    if not states.ndim or not states.size or 0 in states.shape:
        raise ValueError('latent_states must have nonempty axes')
    n = states.shape[-1]
    op = np.eye(n) if operator is None else real_array(operator, 'operator')
    if op.ndim != 2 or op.shape[1] != n or not op.shape[0]:
        raise ValueError('operator must have shape (d, n) with d >= 1')
    sd = vector(
        noise_sd, 'noise_sd', len(op), broadcast = True, nonnegative = True,
    )
    seed = integer(measurement_seed, 'measurement_seed')
    rng = np.random.Generator(np.random.PCG64(seed))
    with np.errstate(over = 'ignore', invalid = 'ignore'):
        signal = checked(states @ op.T, 'readout signal')
        noise = checked(
            rng.standard_normal(signal.shape) * sd, 'measurement noise',
        )
        values = checked(signal + noise, 'observations')
    return ObservationResult(
        signal = frozen_array(signal), measurement_noise = frozen_array(noise),
        values = frozen_array(values),
        metadata = MappingProxyType({
            'operator': frozen_array(op), 'noise_sd': sd,
            'measurement_seed': seed, 'dtype': 'float64',
        }),
    )
