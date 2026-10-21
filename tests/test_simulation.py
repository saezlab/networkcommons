"""Independent numerical and contract checks for cyclic simulation."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from networkcommons.simulation import (
    CyclicSystem, Transfer, intervene, observe, sample_linear_stationary,
    simulate_dynamics, solve_equilibrium,
)


def system(**kwargs):
    values = dict(interaction_matrix = [[0, .2], [.3, 0]], basal = [1, 2])
    values.update(kwargs)
    return CyclicSystem(**values)


def run(model, **kwargs):
    values = dict(initial_state = [0, 0], dt = .01, n_steps = 10, process_seed = 3)
    values.update(kwargs)
    return simulate_dynamics(model, **values)


@pytest.mark.parametrize('transfer', [
    Transfer.linear(),
    Transfer.thresholded_tanh(nonlinear_fraction = 0),
    Transfer.centered_tanh(center = 3, curvature = 0),
])
def test_identity_limits(transfer):
    z = np.array([-1e300, -1, 0, 3, 1e300])
    assert_array_equal(transfer(z), z)
    assert transfer.is_identity
    model = system(transfer = transfer)
    result = solve_equilibrium(model, method = 'linear')
    assert_allclose(result.state, np.linalg.solve(np.eye(2) - model.interaction_matrix, [1, 2]))


def test_transfer_formulas_and_center():
    z = np.array([-2, -.5, 0, .5, 2])
    assert_allclose(Transfer.tanh(gain = 2)(z), np.tanh(2 * z))
    assert_allclose(
        Transfer.thresholded_tanh(gain = 2, threshold = .5, nonlinear_fraction = .7)(z),
        .3 * z + .7 * np.tanh(2 * np.array([-1.5, 0, 0, 0, 1.5])),
    )
    transfer = Transfer.centered_tanh(center = .3, curvature = 8)
    assert transfer(.3) == .3
    derivative = (transfer(.3 + 1e-6) - transfer(.3 - 1e-6)) / 2e-6
    assert derivative == pytest.approx(1, abs = 1e-9)
    model = system(transfer = transfer)
    changed = intervene(model, transfer = {1: Transfer.linear()})
    assert changed.transfer[0] == transfer
    assert changed.transfer[1].is_identity
    assert model.transfer == (transfer, transfer)


@pytest.mark.parametrize('factory,kwargs', [
    (Transfer.tanh, {'gain': 0}), (Transfer.tanh, {'gain': np.inf}),
    (Transfer.tanh, {'gain': 1j}), (Transfer.tanh, {'gain': [1]}),
    (Transfer.thresholded_tanh, {'threshold': -1}),
    (Transfer.thresholded_tanh, {'nonlinear_fraction': 1.1}),
    (Transfer.centered_tanh, {'center': 0, 'curvature': -1}),
])
def test_invalid_transfer(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


def test_system_copies_and_freezes_all_parameters():
    matrix, basal, clamps = np.eye(2), np.ones(2), {0: 2}
    model = CyclicSystem(matrix, basal = basal, clamps = clamps)
    matrix[:] = 4
    basal[:] = 4
    clamps[0] = 10
    assert_array_equal(model.interaction_matrix, np.eye(2))
    assert_array_equal(model.basal, [1, 1])
    assert model.clamps[0] == 2
    for name in ('interaction_matrix', 'basal', 'production_scale', 'production_offset', 'diffusion'):
        array = getattr(model, name)
        with pytest.raises(ValueError):
            array.flat[0] = 7
        with pytest.raises(ValueError):
            array.setflags(write = True)
    with pytest.raises(TypeError):
        model.clamps[0] = 7
    with pytest.raises(FrozenInstanceError):
        model.basal = np.zeros(2)
    with pytest.raises(TypeError):
        model.transfer[0].parameters['gain'] = 7


@pytest.mark.parametrize('kwargs', [
    {'interaction_matrix': []}, {'interaction_matrix': [[1, 2]]},
    {'interaction_matrix': [[0, 1j], [0, 0]]},
    {'basal': 1}, {'basal': [[1], [2]]}, {'basal': [1, np.nan]},
    {'basal': ['1', '2']}, {'production_scale': -1},
    {'production_offset': [1, 2j]}, {'diffusion': np.inf},
    {'diffusion': [[1], [2]]}, {'transfer': [Transfer.linear()]},
    {'transfer': ['linear', 'linear']}, {'clamps': {True: 0}},
    {'clamps': {1.0: 0}}, {'clamps': {-1: 0}}, {'clamps': {2: 0}},
    {'clamps': {0: np.inf}}, {'clamps': {0: 1j}},
])
def test_invalid_system(kwargs):
    with pytest.raises((ValueError, TypeError)):
        system(**kwargs)


def test_intervention_algebra_and_baseline():
    original = system()
    incoming = intervene(original, incoming_scale = {0: 0})
    outgoing = intervene(original, outgoing_scale = {0: 0})
    assert_array_equal(incoming.interaction_matrix, [[0, 0], [.3, 0]])
    assert_array_equal(outgoing.interaction_matrix, [[0, .2], [0, 0]])
    both = intervene(original, incoming_scale = {0: 2}, outgoing_scale = {1: 3}, input_shift = {1: .4})
    assert_allclose(both.interaction_matrix, [[0, 1.2], [.3, 0]])
    assert_allclose(both.basal, [1, 2.4])
    assert_array_equal(original.basal, [1, 2])
    assert_array_equal(original.interaction_matrix, [[0, .2], [.3, 0]])
    soft = intervene(original, production_scale = {0: .4}, production_offset = {0: .06}, diffusion = {0: .02})
    assert_allclose(soft.production_scale, [.4, 1])
    assert_allclose(soft.production_offset, [.06, 0])
    assert_allclose(soft.diffusion, [.02, 0])
    nested = intervene(soft, production_scale = {0: .8}, production_offset = {0: .4}, input_shift = {1: 1})
    assert nested.production_scale[0] == .8
    assert nested.production_offset[0] == .4
    assert nested.basal[1] == 3
    assert_array_equal(intervene(original).interaction_matrix, original.interaction_matrix)


@pytest.mark.parametrize('edit', ['input_shift', 'incoming_scale', 'transfer', 'production_scale', 'production_offset', 'diffusion'])
def test_discarded_receiver_edits_rejected(edit):
    value = Transfer.linear() if edit == 'transfer' else 1
    for model, extra in ((system(clamps = {0: 2}), {}), (system(), {'clamp': {0: 2}})):
        with pytest.raises(ValueError, match = 'clamped receiver'):
            intervene(model, **{edit: {0: value}}, **extra)
    model = intervene(system(clamps = {0: 2}), outgoing_scale = {0: .5}, clamp = {0: 3})
    assert model.clamps[0] == 3
    assert model.interaction_matrix[1, 0] == .15


@pytest.mark.parametrize('kwargs', [
    {'input_shift': {True: 2}}, {'incoming_scale': {0: -1}},
    {'outgoing_scale': {0: 1j}}, {'transfer': {0: 'tanh'}},
    {'diffusion': {0: -1}}, {'production_scale': {0: -1}},
    {'production_offset': {0: np.nan}}, {'clamp': {2: 0}},
])
def test_invalid_interventions(kwargs):
    with pytest.raises(ValueError):
        intervene(system(), **kwargs)


def test_pretransfer_input_and_posttransfer_offset_differ():
    original = system(transfer = Transfer.tanh())
    shifted = run(intervene(original, input_shift = {0: 1}), n_steps = 1)
    offset = run(intervene(original, production_offset = {0: 1}), n_steps = 1)
    assert shifted.final_states[0, 0] == pytest.approx(.01 * np.tanh(2))
    assert offset.final_states[0, 0] == pytest.approx(.01 * (1 + np.tanh(1)))


def test_euler_step_and_linear_equilibrium():
    model = system(production_scale = [2, .5], production_offset = [-1, .4])
    initial = np.array([.3, -.2])
    expected = initial + .01 * (
        np.array([-1, .4]) + np.array([2, .5]) *
        (np.array([[0, .2], [.3, 0]]) @ initial + [1, 2]) - initial
    )
    assert_allclose(run(model, initial_state = initial, n_steps = 1).final_states[0], expected)
    equilibrium = solve_equilibrium(model, method = 'linear')
    assert_allclose(run(model, n_steps = 4000).final_states[0], equilibrium.state, atol = 1e-10)
    fixed = solve_equilibrium(model, initial_state = [0, 0])
    assert_allclose(fixed.state, equilibrium.state, rtol = 1e-7)


def test_skips_constant_production_and_clamped_overflows():
    model = CyclicSystem(
        [[1e308, 1e308], [1e308, 1e308]], basal = [1e308, 1e308],
        transfer = Transfer.tanh(gain = 1e308), production_scale = [0, 1],
        production_offset = [2, 0], clamps = {1: 1e308},
    )
    result = run(model, initial_state = [0, -1e308], n_steps = 1)
    assert_array_equal(result.final_states, [[.02, 1e308]])
    equilibrium = solve_equilibrium(model, method = 'linear')
    assert_array_equal(equilibrium.state, [2, 1e308])
    assert_array_equal(sample_linear_stationary(model, n_samples = 2, process_seed = 3).samples, [[2, 1e308]] * 2)


def test_clamps_initialization_outgoing_and_all_node_cases():
    model = system(clamps = {0: 3}, diffusion = .2)
    result = run(model, initial_state = [99, 0], save_every = 3)
    assert_array_equal(result.initial_states, [[3, 0]])
    assert_array_equal(result.states[:, :, 0], 3)
    quiet = run(system(clamps = {0: 3}), n_steps = 1)
    assert quiet.final_states[0, 1] == pytest.approx(.01 * (2 + .3 * 3))
    all_clamped = system(clamps = {0: 3, 1: -2}, transfer = Transfer.tanh())
    assert_array_equal(run(all_clamped, save_every = 3).states, np.tile([3, -2], (5, 1, 1)))
    assert_array_equal(solve_equilibrium(all_clamped).state, [3, -2])
    stationary = sample_linear_stationary(all_clamped, n_samples = 4, process_seed = 1)
    assert_array_equal(stationary.samples, [[3, -2]] * 4)
    assert_array_equal(stationary.covariance, 0)
    assert stationary.metadata['stability_margin'] is None


def test_trace_shapes_random_streams_and_result_immutability():
    model = system(diffusion = .2)
    result = run(model, initial_state = np.zeros((3, 2)), save_every = 3)
    assert_array_equal(result.times, [0, .03, .06, .09, .1])
    assert result.states.shape == (5, 3, 2)
    assert_array_equal(result.states[0], result.initial_states)
    assert_array_equal(result.states[-1], result.final_states)
    assert_array_equal(result.final_states, run(model, initial_state = np.zeros((3, 2))).final_states)
    assert run(model).states is None
    assert len(run(model, save_every = 2).times) == 6
    assert not np.array_equal(result.final_states[0], result.final_states[1])
    with pytest.raises(ValueError):
        result.final_states[0, 0] = 1
    with pytest.raises(TypeError):
        result.metadata['dt'] = 2
    assert_array_equal(run(system(), process_seed = 1).final_states, run(system(), process_seed = 99).final_states)
    # With no causal edge, clamping node zero cannot change node one's draws.
    unconnected = system(interaction_matrix = np.zeros((2, 2)), diffusion = .3)
    assert_array_equal(
        run(unconnected).final_states[:, 1],
        run(intervene(unconnected, clamp = {0: 3})).final_states[:, 1],
    )


@pytest.mark.parametrize('kwargs', [
    {'dt': 0}, {'dt': np.inf}, {'dt': 1j}, {'dt': 1e308, 'n_steps': 10},
    {'n_steps': True}, {'n_steps': 1.0}, {'n_steps': 0},
    {'process_seed': False}, {'process_seed': -1}, {'process_seed': 3.0},
    {'save_every': 0}, {'save_every': True}, {'save_every': 2.0},
    {'initial_state': [[1], [2]]}, {'initial_state': []},
    {'initial_state': [[1, np.nan]]}, {'initial_state': [0j, 0]},
])
def test_invalid_dynamics(kwargs):
    with pytest.raises(ValueError):
        run(system(), **kwargs)


def test_finite_unstable_paths_and_overflow_diagnostics():
    unstable = CyclicSystem([[2]], basal = [1])
    result = simulate_dynamics(unstable, initial_state = [0], dt = .1, n_steps = 10, process_seed = 0)
    assert result.final_states[0, 0] > 1
    assert 'stationary' not in result.metadata
    assert_array_equal(solve_equilibrium(unstable, method = 'linear').state, [-1])
    with pytest.raises(ValueError, match = 'positive real'):
        sample_linear_stationary(unstable, n_samples = 1, process_seed = 0)
    exploding = CyclicSystem([[1e308]], basal = [0])
    with pytest.raises(FloatingPointError, match = 'node 0.*step 0, time 0.0'):
        simulate_dynamics(exploding, initial_state = [2], dt = 1, n_steps = 1, process_seed = 0)


def test_observation_axes_and_noise_separation():
    latent = run(system(diffusion = .2), save_every = 2).states
    operator = np.array([[1, -1]])
    result = observe(latent, operator = operator, noise_sd = .2, measurement_seed = 4)
    assert result.values.shape == (6, 1, 1)
    assert_allclose(result.signal, latent @ operator.T)
    assert_array_equal(result.values, result.signal + result.measurement_noise)
    other = observe(latent, operator = operator, noise_sd = .2, measurement_seed = 5)
    assert_array_equal(result.signal, other.signal)
    assert not np.array_equal(result.values, other.values)
    assert_array_equal(observe([1, 2], measurement_seed = 1).values, [1, 2])
    assert not result.values.flags.writeable


@pytest.mark.parametrize('states,kwargs', [
    ([], {}), (3, {}), (np.empty((1, 0, 2)), {}), ([1j], {}),
    ([1, 2], {'operator': [[1]]}), ([1, 2], {'operator': []}),
    ([1, 2], {'noise_sd': -1}), ([1, 2], {'noise_sd': [[1, 2]]}),
    ([1, 2], {'measurement_seed': True}),
])
def test_invalid_observation(states, kwargs):
    values = {'measurement_seed': 1, **kwargs}
    with pytest.raises(ValueError):
        observe(states, **values)


def test_exact_linear_reduction_and_stationary_covariance():
    model = CyclicSystem(
        [[.1, .4, .2], [-.2, .1, .3], [.1, .1, 0]],
        basal = [1, 2, 3], production_scale = [.7, 1.2, 1],
        production_offset = [.2, -.1, 0], diffusion = [.3, .2, 5], clamps = {2: 2},
    )
    k = np.eye(2) - np.diag([.7, 1.2]) @ np.array([[.1, .4], [-.2, .1]])
    d = np.array([.2, -.1]) + np.array([.7, 1.2]) * (np.array([1, 2]) + np.array([.2, .3]) * 2)
    stationary = sample_linear_stationary(model, n_samples = 60000, process_seed = 32)
    assert_allclose(stationary.mean[:2], np.linalg.solve(k, d))
    cov = stationary.covariance[:2, :2]
    assert_allclose(k @ cov + cov @ k.T, np.diag([.09, .04]), atol = 1e-14)
    assert stationary.metadata['lyapunov_residual'] < 1e-14
    assert_array_equal(stationary.samples[:, 2], 2)
    assert_array_equal(stationary.covariance[2], 0)
    assert_allclose(stationary.samples.mean(axis = 0), stationary.mean, atol = .004)
    assert_allclose(np.cov(stationary.samples, rowvar = False), stationary.covariance, atol = .001)
    assert_allclose(solve_equilibrium(model, method = 'linear').state, stationary.mean)


@pytest.mark.parametrize('diffusion', [[0, 0], [0, .2]])
def test_degenerate_diffusion_without_jitter(diffusion):
    model = system(interaction_matrix = np.zeros((2, 2)), diffusion = diffusion)
    result = sample_linear_stationary(model, n_samples = 100, process_seed = 1)
    assert_allclose(result.covariance, np.diag(np.array(diffusion) ** 2 / 2), atol = 0)
    assert_array_equal(result.samples[:, 0], 1)
    assert result.metadata['largest_clipped_eigenvalue'] == 0


def test_production_replacement_is_not_clamp():
    original = system(diffusion = .2)
    constant = intervene(original, production_scale = {0: 0}, production_offset = {0: .1})
    soft = sample_linear_stationary(constant, n_samples = 200, process_seed = 5)
    hard = sample_linear_stationary(intervene(original, clamp = {0: .1}), n_samples = 200, process_seed = 5)
    assert soft.mean[0] == .1
    assert soft.covariance[0, 0] == pytest.approx(.2 ** 2 / 2)
    assert hard.covariance[0, 0] == 0
    assert run(constant, n_steps = 1).final_states[0, 0] != .1


def test_reduced_stability_and_nonlinear_eligibility():
    original = system(interaction_matrix = [[2, 0], [0, 0]], transfer = [Transfer.tanh(), Transfer.linear()])
    with pytest.raises(ValueError, match = 'not affine'):
        sample_linear_stationary(original, n_samples = 1, process_seed = 0)
    # Discarded nonlinear/unstable rows do not restrict the reduced operator.
    result = sample_linear_stationary(intervene(original, clamp = {0: 3}), n_samples = 1, process_seed = 0)
    assert_array_equal(result.mean, [3, 2])
    with pytest.raises(np.linalg.LinAlgError):
        solve_equilibrium(system(interaction_matrix = np.eye(2)), method = 'linear')


@pytest.mark.parametrize('kwargs', [
    {'initial_state': None}, {'relaxation': 0}, {'relaxation': 1.1},
    {'max_iterations': 0}, {'max_iterations': True}, {'atol': 0}, {'rtol': -1},
    {'method': 'other'}, {'method': 'linear', 'initial_state': [0, 0]},
    {'method': 'linear', 'relaxation': .5},
    {'method': 'linear', 'max_iterations': 10},
])
def test_invalid_equilibrium(kwargs):
    values = dict(initial_state = [0, 0])
    if kwargs.get('method') == 'linear':
        values = {}
    values.update(kwargs)
    with pytest.raises(ValueError):
        solve_equilibrium(system(), **values)


def test_nonlinear_equilibria_convergence_and_failure():
    model = system(transfer = Transfer.tanh())
    result = solve_equilibrium(model, initial_state = [0, 0])
    assert_allclose(result.state, np.tanh(model.interaction_matrix @ result.state + model.basal), rtol = 1e-8)
    assert result.iterations > 0
    with pytest.raises(RuntimeError, match = 'did not converge'):
        solve_equilibrium(model, initial_state = [0, 0], max_iterations = 1)
    centered = system(transfer = [Transfer.centered_tanh(center = .2, curvature = 8), Transfer.centered_tanh(center = .4, curvature = 8)])
    centered = CyclicSystem(centered.interaction_matrix, basal = np.array([.2, .4]) - centered.interaction_matrix @ [.2, .4], transfer = centered.transfer)
    assert solve_equilibrium(centered, initial_state = [.2, .4]).iterations == 0


@pytest.mark.parametrize('kwargs', [
    {'n_samples': 0}, {'n_samples': True}, {'n_samples': 2.0},
    {'process_seed': -1}, {'process_seed': False}, {'atol': 0}, {'rtol': -1},
])
def test_invalid_stationary(kwargs):
    values = dict(n_samples = 2, process_seed = 1)
    values.update(kwargs)
    with pytest.raises(ValueError):
        sample_linear_stationary(system(), **values)


def test_scalar_finite_time_moments_and_time_step_error():
    gamma, sigma, mu, x0 = .8, .6, 1.5, -.3
    model = CyclicSystem([[.2]], basal = [mu * gamma], diffusion = sigma)
    duration = 2
    continuous_mean = mu + (x0 - mu) * np.exp(-gamma * duration)
    continuous_var = sigma ** 2 / (2 * gamma) * (1 - np.exp(-2 * gamma * duration))
    errors = []
    for dt in (.1, .02):
        steps, paths = round(duration / dt), 50000
        result = simulate_dynamics(model, initial_state = np.full((paths, 1), x0), dt = dt, n_steps = steps, process_seed = 73)
        r = 1 - gamma * dt
        mean = mu + (x0 - mu) * r ** steps
        variance = sigma ** 2 * dt * (1 - r ** (2 * steps)) / (1 - r ** 2)
        empirical = result.final_states[:, 0]
        assert abs(empirical.mean() - mean) < 5 * np.sqrt(variance / paths)
        assert abs(empirical.var() - variance) < 5 * variance * np.sqrt(2 / (paths - 1))
        errors.append(abs(mean - continuous_mean) + abs(variance - continuous_var))
    assert errors[1] < errors[0] / 4
    stationary = sample_linear_stationary(model, n_samples = 50000, process_seed = 8)
    assert_allclose(stationary.mean, [mu])
    assert stationary.covariance[0, 0] == pytest.approx(sigma ** 2 / (2 * gamma))


def test_increasing_duration_at_fixed_dt_approaches_stationary_moments():
    model = CyclicSystem([[0]], basal = [2], diffusion = .4)
    deviations = []
    for steps in (10, 700):
        endpoints = simulate_dynamics(model, initial_state = np.zeros((20000, 1)), dt = .01, n_steps = steps, process_seed = 12).final_states[:, 0]
        deviations.append(abs(endpoints.mean() - 2) + abs(endpoints.var() - .08))
    assert deviations[1] < .01
    assert deviations[1] < deviations[0] / 100


def test_stationary_covariance_rejects_invalid_solver_results(monkeypatch):
    import scipy.linalg
    for bad_covariance, message in (
        (np.diag([-1., 1.]), 'positive semidefinite'),
        (np.eye(2), 'Lyapunov residual'),
    ):
        monkeypatch.setattr(
            scipy.linalg, 'solve_continuous_lyapunov',
            lambda k, d, value = bad_covariance: value,
        )
        with pytest.raises(RuntimeError, match = message):
            sample_linear_stationary(system(), n_samples = 1, process_seed = 0)
    # A declared roundoff correction changes the sampling law and diagnostics.
    monkeypatch.setattr(
        scipy.linalg, 'solve_continuous_lyapunov',
        lambda k, d: np.diag([-1e-14, 0]),
    )
    result = sample_linear_stationary(system(), n_samples = 4, process_seed = 0)
    assert result.metadata['largest_clipped_eigenvalue'] == 1e-14
    assert result.metadata['lyapunov_residual'] == 0
    assert_array_equal(result.covariance, 0)
    assert_array_equal(result.samples, np.tile(result.mean, (4, 1)))


def test_all_clamped_operations_still_validate_arguments():
    model = system(clamps = {0: 1, 1: 2})
    for kwargs in ({'relaxation': -1}, {'initial_state': [0]}, {'max_iterations': True}):
        with pytest.raises(ValueError):
            solve_equilibrium(model, **kwargs)
    with pytest.raises(ValueError):
        sample_linear_stationary(model, n_samples = True, process_seed = 1)


def test_example_artifacts_and_reproducibility(tmp_path):
    import json
    import runpy
    from pathlib import Path
    path = Path(__file__).parents[1] / 'examples/cyclic_simulator.py'
    generate = runpy.run_path(str(path))['generate']
    first, second = tmp_path / 'first', tmp_path / 'second'
    names = generate(first, n_samples = 8, master_seed = 37)
    assert len(names) == 5
    generate(second, n_samples = 8, master_seed = 37)
    for name in names:
        assert (first / f'{name}.json').read_bytes() == (second / f'{name}.json').read_bytes()
        metadata = json.loads((first / f'{name}.json').read_text())
        assert metadata['source_sha256']
        with np.load(first / f'{name}.npz', allow_pickle = False) as a, np.load(second / f'{name}.npz', allow_pickle = False) as b:
            for key in a.files:
                assert_array_equal(a[key], b[key])
            for condition in metadata['conditions']:
                assert condition['process_seed'] != condition['measurement_seed']
                prefix = condition['prefix']
                assert a[prefix + '_latent'].shape == (8, 3)
                assert a[prefix + '_observed'].shape == (8, 3)
                for i, value in condition['model']['clamps'].items():
                    assert_array_equal(a[prefix + '_latent'][:, int(i)], value)


def test_large_finite_covariance_has_finite_diagnostics():
    model = CyclicSystem([[0]], basal = [0], diffusion = 1e100)
    result = sample_linear_stationary(model, n_samples = 10, process_seed = 2)
    assert result.covariance[0, 0] == pytest.approx(5e199)
    assert np.isfinite(result.metadata['lyapunov_residual'])
    assert np.isfinite(result.samples).all()
