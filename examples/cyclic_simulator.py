"""
Generate five reproducible synthetic experiments with the public API.

Run from the repository: uv run --locked python examples/cyclic_simulator.py
Generated arrays and metadata go to build/simulation and are not source files.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np

from networkcommons.simulation import (
    CyclicSystem, Transfer, intervene, observe, sample_linear_stationary,
    simulate_dynamics, solve_equilibrium,
)


A = np.array([[0, .2, 0], [0, 0, .3], [.1, 0, 0]])
CENTER = np.array([.2, .3, .4])


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, 'items'):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def model_record(model):
    record = {
        name: getattr(model, name)
        for name in ('interaction_matrix', 'basal', 'production_scale',
                     'production_offset', 'diffusion', 'clamps')
    }
    record['transfer'] = [
        {'kind': t.kind, 'parameters': t.parameters} for t in model.transfer
    ]
    return json_value(record)


def experiments():
    """Explicit experiment choices, separate from the numerical core."""
    linear = CyclicSystem(A, basal = np.ones(3), diffusion = .1)
    centered = lambda k, rho: CyclicSystem(
        rho * A, basal = CENTER - rho * A @ CENTER,
        transfer = tuple(Transfer.centered_tanh(center = c, curvature = k) for c in CENTER),
    )
    return [
        ('equilibria', [
            ('linear_control', centered(0, 1), 'linear', {}),
            ('linear_shift', intervene(centered(0, 1), input_shift = {0: .2}), 'linear', {'input_shift': {0: .2}}),
            ('linear_hard', intervene(centered(0, 1), clamp = {0: 0}), 'linear', {'clamp': {0: 0}}),
            ('saturating_control', centered(8, 1), 'fixed_point', {}),
            ('saturating_shift', intervene(centered(8, 1), input_shift = {0: .2}), 'fixed_point', {'input_shift': {0: .2}}),
            ('saturating_hard', intervene(centered(8, 1), clamp = {0: 0}), 'fixed_point', {'clamp': {0: 0}}),
        ]),
        ('linear_stochastic', [
            ('control', linear, 'stationary', {}),
            ('replacement', intervene(linear, production_scale = {0: 0}, production_offset = {0: .1}, diffusion = {0: .001}), 'stationary', {'production_scale': {0: 0}, 'production_offset': {0: .1}, 'diffusion': {0: .001}}),
        ]),
        ('coupling_saturation', [
            (f'rho_{rho}_k_{k}_{label}', model, method, edit)
            for rho in (.5, 1., 1.5)
            for k in (0., 8.)
            for label, model, edit in (
                ('control', centered(k, rho), {}),
                ('hard', intervene(centered(k, rho), clamp = {0: 0}), {'clamp': {0: 0}}),
            )
            for method in (('linear' if k == 0 else 'fixed_point'),)
        ]),
        ('process_noise', [
            (f'diffusion_{sigma}', CyclicSystem(A, basal = CENTER - A @ CENTER, diffusion = sigma), 'finite_time', {'diffusion_all': sigma})
            for sigma in (0., .05, .2)
        ]),
        ('production_replacement', [
            (f'fraction_{fraction}', intervene(linear, production_scale = {0: 1 - fraction}, production_offset = {0: fraction * .1}, diffusion = {0: .02}), 'stationary', {'replacement_fraction': fraction, 'replacement_level': .1, 'diffusion': {0: .02}})
            for fraction in (0., .3, .6, 1.)
        ] + [('hard', intervene(linear, clamp = {0: .1}), 'stationary', {'clamp': {0: .1}})]),
    ]


def generate(output, *, n_samples = 250, master_seed = 20261021):
    """Persist numerical arrays plus complete per-condition descriptions."""
    output = Path(output)
    output.mkdir(parents = True, exist_ok = True)
    root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__), *sorted((root / 'networkcommons/simulation').glob('*.py'))]
    sources = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    try:
        revision = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text = True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    all_experiments = experiments()
    total = sum(len(conditions) for _, conditions in all_experiments)
    condition_streams = iter(np.random.SeedSequence(master_seed).spawn(total))
    for experiment, conditions in all_experiments:
        arrays, records = {}, []
        for index, (label, model, method, intervention) in enumerate(conditions):
            stream = next(condition_streams)
            process_child, measurement_child = stream.spawn(2)
            process_seed = int(process_child.generate_state(1, dtype = np.uint64)[0])
            measurement_seed = int(measurement_child.generate_state(1, dtype = np.uint64)[0])
            initial = None
            if method in ('linear', 'fixed_point'):
                kwargs = {'initial_state': CENTER} if method == 'fixed_point' else {}
                result = solve_equilibrium(model, method = method, **kwargs)
                latent = np.repeat(result.state[None, :], n_samples, axis = 0)
                reference, covariance = result.state, np.zeros((3, 3))
                diagnostics = {'free_residual': result.free_residual, 'clamp_residual': result.clamp_residual, 'iterations': result.iterations}
                output_kind = 'structural_equilibrium'
                initial = CENTER if method == 'fixed_point' else None
            elif method == 'stationary':
                result = sample_linear_stationary(model, n_samples = n_samples, process_seed = process_seed)
                latent, reference, covariance = result.samples, result.mean, result.covariance
                diagnostics = dict(result.metadata)
                output_kind = 'analytic_stationary'
            else:
                initial = np.repeat(CENTER[None, :], n_samples, axis = 0)
                result = simulate_dynamics(model, initial_state = initial, dt = .01, n_steps = 2000, process_seed = process_seed)
                latent = result.final_states
                # These are finite-time paths, with a stationary reference for comparison.
                target = sample_linear_stationary(model, n_samples = 1, process_seed = process_seed)
                reference, covariance = target.mean, target.covariance
                diagnostics = {
                    'reference_kind': 'analytic_stationary',
                    'finite_time_mean_max_error': float(np.max(np.abs(latent.mean(axis = 0) - reference))),
                    'mean_standard_error': np.sqrt(np.diag(covariance) / n_samples),
                    'validation': 'Analytic finite-time moments and fixed-duration time-step refinement are tested in tests/test_simulation.py. These endpoints are not certified stationary.',
                }
                output_kind = 'finite_time_endpoint'
            observations = observe(latent, operator = np.eye(3), noise_sd = .01, measurement_seed = measurement_seed)
            prefix = f'condition_{index}'
            for name, value in (
                ('latent', latent), ('signal', observations.signal),
                ('measurement_noise', observations.measurement_noise),
                ('observed', observations.values), ('reference_mean', reference),
                ('reference_covariance', covariance),
                ('replicate', np.arange(n_samples)),
            ):
                arrays[f'{prefix}_{name}'] = value
            if initial is not None:
                arrays[f'{prefix}_initial_state'] = initial
            # Whole conditions are held out. No preprocessing is fitted here.
            split = ('train', 'validation', 'test')[index % 3]
            arrays[f'{prefix}_split'] = np.full(n_samples, split)
            arrays[f'{prefix}_condition'] = np.full(n_samples, label)
            records.append({
                'prefix': prefix, 'condition': label, 'split': split,
                'intervention': intervention, 'model': model_record(model),
                'output_kind': output_kind, 'numerics': dict(result.metadata),
                'diagnostics': diagnostics, 'readout': dict(observations.metadata),
                'initial_state_key': None if initial is None else f'{prefix}_initial_state',
                'process_seed': process_seed, 'measurement_seed': measurement_seed,
                'seed_derivation': {
                    'master_seed': master_seed,
                    'condition_spawn_key': stream.spawn_key,
                    'process_spawn_key': process_child.spawn_key,
                    'measurement_spawn_key': measurement_child.spawn_key,
                    'integer_derivation': 'child.generate_state(1, dtype=uint64)[0]',
                },
            })
        np.savez_compressed(output / f'{experiment}.npz', **arrays)
        record = {
            'experiment': experiment, 'variable_order': ['x0', 'x1', 'x2'],
            'equations': 'F=q+p*phi(A@x+b); dX=(F-X)dt+diag(sigma)dW; clamps replace equations; Y=P@X+tau*epsilon',
            'state_domain': 'normalized real values, including negative states',
            'stability': 'All baseline A families have spectral norm < 1. Centered transfers have Lipschitz constant <= 1. Exact stationary calls additionally check reduced stability.',
            'evaluation': 'Compare held-out condition means or distributions with the recorded references. Individual latent samples are not population means.',
            'controls': 'Coupling/saturation control means share CENTER and J=rho*A-I is fixed across curvature. Noise sweep fixes graph, initial states, duration, dt and measurement noise. Replacement sweep fixes target production level and process amplitude, with the hard clamp separately labeled.',
            'split_policy': 'Condition index modulo three. Illustrative only. No fitting, scaling or normalization uses evaluation references.',
            'versions': {name: version(name) for name in ('networkcommons', 'numpy', 'scipy')},
            'python': platform.python_version(), 'revision': revision,
            'source_sha256': sources, 'conditions': records,
        }
        (output / f'{experiment}.json').write_text(json.dumps(json_value(record), indent = 2, allow_nan = False) + '\n')
    return [name for name, _ in all_experiments]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--output', type = Path, default = Path('build/simulation'))
    parser.add_argument('--n-samples', type = int, default = 250)
    parser.add_argument('--seed', type = int, default = 20261021)
    args = parser.parse_args()
    if args.n_samples < 1 or args.seed < 0:
        parser.error('n-samples must be positive and seed must be nonnegative')
    for name in generate(args.output, n_samples = args.n_samples, master_seed = args.seed):
        print(name)
