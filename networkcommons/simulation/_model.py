"""
Declarative mechanisms, validation, and immutable intervention construction.
"""

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike


def real_array(value, name):
    """Copy a finite real numerical array without discarding complex parts."""
    raw = np.asarray(value)
    if raw.dtype.kind not in 'biuf':
        raise ValueError(f'{name} must contain real numbers')
    array = np.array(raw, dtype = np.float64, copy = True)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite')
    return array


def scalar(value, name, *, minimum = None, positive = False):
    array = real_array(value, name)
    if array.ndim != 0:
        raise ValueError(f'{name} must be scalar')
    result = float(array)
    if minimum is not None and result < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    if positive and result <= 0:
        raise ValueError(f'{name} must be positive')
    return result


def integer(value, name, *, minimum = 0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer),
    ):
        raise ValueError(f'{name} must be an integer')
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return int(value)


def frozen_array(value):
    # Immutable backing bytes also prevent callers from re-enabling writes.
    array = np.ascontiguousarray(value, dtype = np.float64)
    return np.frombuffer(
        array.tobytes(), dtype = np.float64,
    ).reshape(array.shape)


def vector(value, name, n, *, broadcast = False, nonnegative = False):
    array = real_array(value, name)
    if broadcast and array.ndim == 0:
        array = np.full(n, float(array))
    if array.shape != (n,):
        raise ValueError(f'{name} must have shape ({n},)')
    if nonnegative and (array < 0).any():
        raise ValueError(f'{name} must be nonnegative')
    return frozen_array(array)


def edits(mapping, name, n):
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise ValueError(f'{name} must be a mapping')
    result = {}
    for key, value in mapping.items():
        index = integer(key, f'{name} index')
        if index >= n:
            raise ValueError(f'{name} index {index} is out of range')
        result[index] = value
    return result


@dataclass(frozen = True, init = False)
class Transfer:
    """
    Immutable scalar transfer, constructed with the four named factories.

    Attributes:
        kind: Serializable transfer name.
        parameters: Read-only scalar parameter mapping.
    """

    kind: str
    parameters: Mapping[str, float]

    def __init__(self, kind, **parameters):
        expected = {
            'linear': (),
            'tanh': ('gain',),
            'thresholded_tanh': ('gain', 'threshold', 'nonlinear_fraction'),
            'centered_tanh': ('center', 'curvature'),
        }
        if kind not in expected or set(parameters) != set(expected[kind]):
            raise ValueError('Unknown transfer kind or parameters')
        normalized = {
            key: scalar(value, key) for key, value in parameters.items()
        }
        if 'gain' in normalized and normalized['gain'] <= 0:
            raise ValueError('gain must be positive')
        for key in ('threshold', 'curvature'):
            if key in normalized and normalized[key] < 0:
                raise ValueError(f'{key} must be nonnegative')
        fraction = normalized.get('nonlinear_fraction', 0)
        if not 0 <= fraction <= 1:
            raise ValueError('nonlinear_fraction must be between zero and one')
        object.__setattr__(self, 'kind', kind)
        object.__setattr__(self, 'parameters', MappingProxyType(normalized))

    @classmethod
    def linear(cls) -> 'Transfer':
        """Return the identity transfer."""
        return cls('linear')

    @classmethod
    def tanh(cls, *, gain: float = 1.0) -> 'Transfer':
        """Return tanh(gain * input), with positive gain."""
        return cls('tanh', gain = gain)

    @classmethod
    def thresholded_tanh(
        cls, *, gain: float = 1.0, threshold: float = 0.0,
        nonlinear_fraction: float = 1.0,
    ) -> 'Transfer':
        """Mix identity with continuous thresholded saturation."""
        return cls(
            'thresholded_tanh', gain = gain, threshold = threshold,
            nonlinear_fraction = nonlinear_fraction,
        )

    @classmethod
    def centered_tanh(
        cls, *, center: float, curvature: float,
    ) -> 'Transfer':
        """Preserve the value and unit slope at center."""
        return cls('centered_tanh', center = center, curvature = curvature)

    @property
    def is_identity(self):
        return (
            self.kind == 'linear' or
            (self.kind == 'thresholded_tanh' and
             self.parameters['nonlinear_fraction'] == 0) or
            (self.kind == 'centered_tanh' and
             self.parameters['curvature'] == 0)
        )

    def __call__(self, z):
        """Evaluate the declared transfer, including exact identity limits."""
        p = self.parameters
        if self.is_identity:
            return z
        if self.kind == 'tanh':
            return np.tanh(p['gain'] * z)
        if self.kind == 'thresholded_tanh':
            r = np.sign(z) * np.maximum(np.abs(z) - p['threshold'], 0)
            nonlinear = np.tanh(p['gain'] * r)
            a = p['nonlinear_fraction']
            return nonlinear if a == 1 else (1 - a) * z + a * nonlinear
        k, c = p['curvature'], p['center']
        return c + np.tanh(k * (z - c)) / k


@dataclass(frozen = True, init = False, eq = False)
class CyclicSystem:
    """
    Continuous real-state model with unit decay and independent diffusion.

    Args:
        interaction_matrix: Square matrix, receiver rows and source columns.
        basal: Length-n input vector applied before transfer evaluation.
        transfer: Shared transfer or one transfer per coordinate.
        production_scale: Nonnegative scalar or length-n vector.
        production_offset: Scalar or length-n vector after the transfer.
        diffusion: Nonnegative scalar or length-n process amplitude vector.
        clamps: Mapping of zero-based node indices to fixed state values.

    All arrays are copied float64 values with immutable storage.
    """

    interaction_matrix: np.ndarray
    basal: np.ndarray
    transfer: tuple[Transfer, ...]
    production_scale: np.ndarray
    production_offset: np.ndarray
    diffusion: np.ndarray
    clamps: Mapping[int, float]

    def __init__(
        self, interaction_matrix: ArrayLike, *, basal: ArrayLike,
        transfer: Transfer | Sequence[Transfer] = Transfer.linear(),
        production_scale: float | ArrayLike = 1.0,
        production_offset: float | ArrayLike = 0.0,
        diffusion: float | ArrayLike = 0.0,
        clamps: Mapping[int, float] | None = None,
    ) -> None:
        matrix = real_array(interaction_matrix, 'interaction_matrix')
        if matrix.ndim != 2 or not matrix.shape[0] or (
            matrix.shape[0] != matrix.shape[1]
        ):
            raise ValueError('interaction_matrix must be nonempty and square')
        n = matrix.shape[0]
        transfers = (
            (transfer,) * n
            if isinstance(transfer, Transfer) else tuple(transfer)
        )
        if len(transfers) != n or not all(
            isinstance(t, Transfer) for t in transfers
        ):
            raise ValueError('transfer must contain exactly n Transfer values')
        values = {
            'interaction_matrix': frozen_array(matrix),
            'basal': vector(basal, 'basal', n),
            'transfer': transfers,
            'clamps': MappingProxyType({
                i: scalar(v, 'clamp')
                for i, v in edits(clamps, 'clamps', n).items()
            }),
        }
        for name, value, nonnegative in (
            ('production_scale', production_scale, True),
            ('production_offset', production_offset, False),
            ('diffusion', diffusion, True),
        ):
            values[name] = vector(
                value, name, n, broadcast = True, nonnegative = nonnegative,
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)


def intervene(
    system: CyclicSystem, *,
    input_shift: Mapping[int, float] | None = None,
    incoming_scale: Mapping[int, float] | None = None,
    outgoing_scale: Mapping[int, float] | None = None,
    transfer: Mapping[int, Transfer] | None = None,
    production_scale: Mapping[int, float] | None = None,
    production_offset: Mapping[int, float] | None = None,
    diffusion: Mapping[int, float] | None = None,
    clamp: Mapping[int, float] | None = None,
) -> CyclicSystem:
    """
    Return an edited model without mutating the baseline.

    Input shifts add to basal input. Incoming and outgoing scales multiply
    receiver rows and source columns. Other edits replace effective values.
    Receiver edits on resulting clamps fail. Outgoing edits remain valid.
    """
    n = len(system.basal)
    changes = {
        name: edits(value, name, n)
        for name, value in (
            ('input_shift', input_shift), ('incoming_scale', incoming_scale),
            ('outgoing_scale', outgoing_scale), ('transfer', transfer),
            ('production_scale', production_scale),
            ('production_offset', production_offset),
            ('diffusion', diffusion), ('clamp', clamp),
        )
    }
    clamps = dict(system.clamps)
    clamps.update(changes['clamp'])
    for name, values in changes.items():
        discarded = set(values) & set(clamps)
        if name not in ('outgoing_scale', 'clamp') and discarded:
            raise ValueError(f'{name} edits a clamped receiver')
        for i, value in list(values.items()):
            if name == 'transfer':
                if not isinstance(value, Transfer):
                    raise ValueError('transfer edits require Transfer values')
            else:
                minimum = 0 if name in (
                    'incoming_scale', 'outgoing_scale',
                    'production_scale', 'diffusion',
                ) else None
                values[i] = scalar(value, name, minimum = minimum)
    matrix = system.interaction_matrix.copy()
    rows, cols = np.ones(n), np.ones(n)
    for i, v in changes['incoming_scale'].items():
        rows[i] = v
    for i, v in changes['outgoing_scale'].items():
        cols[i] = v
    # Multiplication is sequential to avoid overflow in an irrelevant r*s.
    with np.errstate(over = 'ignore', invalid = 'ignore'):
        matrix = (rows[:, None] * matrix) * cols[None, :]
        basal = system.basal.copy()
        for i, v in changes['input_shift'].items():
            basal[i] += v
    transfers = list(system.transfer)
    for i, v in changes['transfer'].items():
        transfers[i] = v
    kwargs = {}
    for name in ('production_scale', 'production_offset', 'diffusion'):
        value = getattr(system, name).copy()
        for i, v in changes[name].items():
            value[i] = v
        kwargs[name] = value
    return CyclicSystem(
        matrix, basal = basal, transfer = transfers, clamps = clamps, **kwargs,
    )


def free_indices(system):
    return np.array([
        i for i in range(len(system.basal)) if i not in system.clamps
    ], dtype = int)


def project(system, states):
    for i, value in system.clamps.items():
        states[..., i] = value


def checked(value, context, *, step = None, time = None):
    bad = np.argwhere(~np.isfinite(value))
    if len(bad):
        location = tuple(int(i) for i in bad[0])
        raise FloatingPointError(
            f'Nonfinite {context} at step {step}, time {time}, '
            f'path/node index {location}',
        )
    return value


def production(system, states, *, step = None, time = None):
    """Evaluate only free mechanisms and skip regulation when p is zero."""
    result = np.zeros_like(states)
    with np.errstate(over = 'ignore', invalid = 'ignore', divide = 'ignore'):
        for i in free_indices(system):
            if system.production_scale[i] == 0:
                result[..., i] = system.production_offset[i]
                continue
            z = states @ system.interaction_matrix[i] + system.basal[i]
            checked(
                z[..., None], f'input for node {i}',
                step = step, time = time,
            )
            phi = system.transfer[i](z)
            checked(
                phi[..., None], f'transfer for node {i}',
                step = step, time = time,
            )
            result[..., i] = (
                system.production_offset[i] + system.production_scale[i] * phi
            )
    return checked(result, 'production', step = step, time = time)
