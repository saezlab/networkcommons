"""
Synthetic continuous-state data with feedback and explicit interventions.

Finite-time dynamics, structural equilibria, and attracting affine stationary
laws are separate operations over the same immutable mechanism definition.
"""

from ._model import CyclicSystem, Transfer, intervene
from ._dynamics import (
    DynamicsResult, ObservationResult, observe, simulate_dynamics,
)
from ._exact import (
    EquilibriumResult, StationaryResult, sample_linear_stationary,
    solve_equilibrium,
)

__all__ = [
    'CyclicSystem', 'Transfer', 'intervene', 'simulate_dynamics', 'observe',
    'solve_equilibrium', 'sample_linear_stationary', 'DynamicsResult',
    'ObservationResult', 'EquilibriumResult', 'StationaryResult',
]
