#!/usr/bin/env python

#
# This file is part of the `networkcommons` Python module
#
# Copyright 2024
# Heidelberg University Hospital
#
# File author(s): Saez Lab (omnipathdb@gmail.com)
#
# Distributed under the GPLv3 license
# See the file `LICENSE` or read a copy at
# https://www.gnu.org/licenses/gpl-3.0.txt
#

"""
Perturbation-response prediction methods.
"""

from __future__ import annotations

__all__ = [
    'network_to_perturbation_table',
    'split_perturbation_data',
    'evaluate_predictions',
    'run_mean_response_baseline',
    'run_ridge_baseline',
    'run_lembas_rnn',
]

import typing as t

import networkx as nx
import numpy as np
import pandas as pd

from networkcommons._session import _log

try:
    import torch
except ImportError:
    torch = None


def _as_dataframe(data, name: str) -> pd.DataFrame:

    if isinstance(data, pd.DataFrame):
        return data.copy()

    raise TypeError(f'`{name}` must be a pandas DataFrame.')


def _align_samples(
        perturbations: pd.DataFrame,
        readouts: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    shared = perturbations.index.intersection(readouts.index)

    if shared.empty:
        raise ValueError('Perturbation and readout tables have no shared rows.')

    return perturbations.loc[shared], readouts.loc[shared]


def _edge_sign(value, unknown_value: float = 0.1) -> float:

    if value is None or pd.isna(value):
        return unknown_value

    if isinstance(value, str):
        value_lower = value.strip().lower()

        if value_lower in {'+', '1', 'activation', 'activating', 'stimulation'}:
            return 1.0

        if value_lower in {'-', '-1', 'inhibition', 'inhibiting'}:
            return -1.0

        return unknown_value

    if value > 0:
        return 1.0

    if value < 0:
        return -1.0

    return unknown_value


def _infer_edge_sign(data: dict, sign_attr: str, unknown_value: float) -> float:

    for attr in (sign_attr, 'mode_of_action', 'interaction', 'weight'):
        if attr in data:
            return _edge_sign(data[attr], unknown_value)

    return unknown_value


def network_to_perturbation_table(
        network: nx.DiGraph,
        sign_attr: str = 'sign',
        source_col: str = 'source',
        target_col: str = 'target',
        moa_col: str = 'mode_of_action',
        unknown_value: float = 0.1,
    ) -> pd.DataFrame:
    """
    Convert a NetworkX graph to the edge table expected by LEMBAS-like models.

    Args:
        network: Directed prior knowledge network.
        sign_attr: Edge attribute containing activation/inhibition signs.
        source_col: Output source column name.
        target_col: Output target column name.
        moa_col: Output mode-of-action column name.
        unknown_value: Numeric value used for unknown mode of action.

    Returns:
        Edge table with source, target and mode-of-action columns.
    """

    if not isinstance(network, nx.DiGraph):
        raise TypeError('`network` must be a networkx.DiGraph.')

    records = [
        {
            source_col: source,
            target_col: target,
            moa_col: _infer_edge_sign(data, sign_attr, unknown_value),
        }
        for source, target, data in network.edges(data=True)
    ]

    return pd.DataFrame.from_records(
        records,
        columns=[source_col, target_col, moa_col],
    )


def split_perturbation_data(
        perturbations: pd.DataFrame,
        readouts: pd.DataFrame,
        test_size: float | int = 0.2,
        seed: int | None = 888,
        shuffle: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split matched perturbation-response tables into train and test sets.

    Args:
        perturbations: Samples by perturbation features.
        readouts: Samples by response readouts.
        test_size: Fraction or absolute number of samples for testing.
        seed: Random seed used when shuffling samples.
        shuffle: Whether to shuffle before splitting.

    Returns:
        ``X_train, X_test, y_train, y_test``.
    """

    perturbations = _as_dataframe(perturbations, 'perturbations')
    readouts = _as_dataframe(readouts, 'readouts')
    perturbations, readouts = _align_samples(perturbations, readouts)

    n_samples = len(perturbations)

    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError('Float `test_size` must be between 0 and 1.')

        n_test = int(np.ceil(n_samples * test_size))

    elif isinstance(test_size, int):
        n_test = test_size

    else:
        raise TypeError('`test_size` must be a float or integer.')

    if n_test <= 0 or n_test >= n_samples:
        raise ValueError('Test split must contain at least one sample and '
                         'leave at least one training sample.')

    order = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)

    test_idx = order[:n_test]
    train_idx = order[n_test:]

    return (
        perturbations.iloc[train_idx],
        perturbations.iloc[test_idx],
        readouts.iloc[train_idx],
        readouts.iloc[test_idx],
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> float:

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def evaluate_predictions(
        readouts: pd.DataFrame,
        predictions: pd.DataFrame,
        axis: t.Literal['readout', 'condition'] = 'readout',
    ) -> pd.DataFrame:
    """
    Calculate simple predictive metrics for perturbation-response models.

    Args:
        readouts: Observed response matrix (conditions × readouts).
        predictions: Predicted response matrix (conditions × readouts).
        axis: Dimension along which to compute per-element metrics.
            ``'readout'`` (default) returns one row per TF/readout, measuring
            how well each output node is predicted across all conditions.
            ``'condition'`` returns one row per sample, measuring how well each
            experimental condition is predicted across all readouts.
            Both modes append an ``__all__`` summary row computed over all
            elements jointly.

    Returns:
        DataFrame with columns ``mse``, ``mae``, ``pearson``, indexed by
        readout name (``axis='readout'``) or condition name
        (``axis='condition'``).

    Raises:
        ValueError: If ``axis`` is not ``'readout'`` or ``'condition'``, or if
            readouts and predictions share no rows/columns.
    """

    if axis not in ('readout', 'condition'):
        raise ValueError(f"`axis` must be 'readout' or 'condition', got {axis!r}")

    readouts = _as_dataframe(readouts, 'readouts')
    predictions = _as_dataframe(predictions, 'predictions')

    shared_rows = readouts.index.intersection(predictions.index)
    shared_cols = readouts.columns.intersection(predictions.columns)

    if shared_rows.empty or shared_cols.empty:
        raise ValueError('Readouts and predictions must share rows and columns.')

    y_true = readouts.loc[shared_rows, shared_cols]
    y_pred = predictions.loc[shared_rows, shared_cols]

    records = []
    index_name = axis  # 'readout' or 'condition'

    if axis == 'readout':
        for label in shared_cols:
            err = y_pred[label].to_numpy() - y_true[label].to_numpy()
            records.append({
                index_name: label,
                'mse': float(np.mean(err ** 2)),
                'mae': float(np.mean(np.abs(err))),
                'pearson': _pearson(y_true[label].to_numpy(), y_pred[label].to_numpy()),
            })
    else:
        for label in shared_rows:
            err = y_pred.loc[label].to_numpy() - y_true.loc[label].to_numpy()
            records.append({
                index_name: label,
                'mse': float(np.mean(err ** 2)),
                'mae': float(np.mean(np.abs(err))),
                'pearson': _pearson(y_true.loc[label].to_numpy(), y_pred.loc[label].to_numpy()),
            })

    flat_err = y_pred.to_numpy().ravel() - y_true.to_numpy().ravel()
    records.append({
        index_name: '__all__',
        'mse': float(np.mean(flat_err ** 2)),
        'mae': float(np.mean(np.abs(flat_err))),
        'pearson': _pearson(y_true.to_numpy().ravel(), y_pred.to_numpy().ravel()),
    })

    return pd.DataFrame.from_records(records).set_index(index_name)


def run_mean_response_baseline(
        readouts_train: pd.DataFrame,
        readouts_eval: pd.DataFrame | None = None,
    ) -> dict[str, t.Any]:
    """
    Predict each readout by its mean value in the training set.
    """

    readouts_train = _as_dataframe(readouts_train, 'readouts_train')

    if readouts_eval is None:
        eval_index = readouts_train.index
    else:
        readouts_eval = _as_dataframe(readouts_eval, 'readouts_eval')
        eval_index = readouts_eval.index

    means = readouts_train.mean(axis=0)
    predictions = pd.DataFrame(
        np.repeat(means.to_numpy()[None, :], len(eval_index), axis=0),
        index=eval_index,
        columns=readouts_train.columns,
    )

    return {
        'predictions': predictions,
        'readout_mean': means,
    }


def _design_matrix(features: pd.DataFrame, fit_intercept: bool) -> np.ndarray:

    matrix = features.to_numpy(dtype=float)

    if fit_intercept:
        matrix = np.column_stack([np.ones(len(features)), matrix])

    return matrix


def run_ridge_baseline(
        perturbations_train: pd.DataFrame,
        readouts_train: pd.DataFrame,
        perturbations_eval: pd.DataFrame | None = None,
        alpha: float = 1.0,
        fit_intercept: bool = True,
    ) -> dict[str, t.Any]:
    """
    Fit a closed-form ridge regression perturbation-response baseline.
    """

    if alpha < 0:
        raise ValueError('`alpha` must be non-negative.')

    perturbations_train = _as_dataframe(
        perturbations_train,
        'perturbations_train',
    )
    readouts_train = _as_dataframe(readouts_train, 'readouts_train')
    perturbations_train, readouts_train = _align_samples(
        perturbations_train,
        readouts_train,
    )

    if perturbations_eval is None:
        perturbations_eval = perturbations_train
    else:
        perturbations_eval = _as_dataframe(
            perturbations_eval,
            'perturbations_eval',
        )
        perturbations_eval = perturbations_eval.loc[
            :,
            perturbations_train.columns,
        ]

    x_train = _design_matrix(perturbations_train, fit_intercept)
    x_eval = _design_matrix(perturbations_eval, fit_intercept)
    y_train = readouts_train.to_numpy(dtype=float)

    penalty = np.eye(x_train.shape[1]) * alpha

    if fit_intercept:
        penalty[0, 0] = 0

    lhs = x_train.T @ x_train + penalty
    rhs = x_train.T @ y_train

    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs

    pred_eval = x_eval @ beta
    pred_train = x_train @ beta

    predictions = pd.DataFrame(
        pred_eval,
        index=perturbations_eval.index,
        columns=readouts_train.columns,
    )

    offset = 1 if fit_intercept else 0
    coefficients = pd.DataFrame(
        beta[offset:, :],
        index=perturbations_train.columns,
        columns=readouts_train.columns,
    )
    intercept = pd.Series(
        beta[0, :] if fit_intercept else np.zeros(readouts_train.shape[1]),
        index=readouts_train.columns,
        name='intercept',
    )

    return {
        'predictions': predictions,
        'coefficients': coefficients,
        'intercept': intercept,
        'train_mse': float(np.mean((pred_train - y_train) ** 2)),
        'alpha': alpha,
    }


def _torch_dtype(dtype):

    if isinstance(dtype, str):
        try:
            return getattr(torch, dtype)
        except AttributeError as exc:
            raise ValueError(f'Unknown torch dtype `{dtype}`.') from exc

    return dtype


def _torch_device(device: str):

    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return torch.device(device)


def _mml_activation(x, leak: float = 0.01):
    """Michaelis-Menten-like activation matching the original LEMBAS paper.

    Maps inputs to [0, 1] for positive values with a leaky linear region
    below zero. This is the default activation in LEMBAS and LEMBAS-GPU.
    """
    fx = torch.nn.functional.leaky_relu(x, negative_slope=leak)
    shifted = 0.5 * (fx - 0.5)
    mask = shifted.lt(0.0)
    safe = fx + 10.0 * mask  # avoid division by zero
    right = 0.5 + shifted / safe
    return mask * (fx - right) + right


def _make_lembas_model():

    class LembasRNN(torch.nn.Module):

        def __init__(
                self,
                n_nodes: int,
                source_idx: np.ndarray,
                target_idx: np.ndarray,
                edge_signs: np.ndarray,
                input_idx: np.ndarray,
                output_idx: np.ndarray,
                n_steps: int,
                tolerance: float,
                activation: str,
                leak: float,
                dtype,
                device,
                learn_input_scale: bool,
                input_scale_init: float,
            ):

            super().__init__()
            self.n_nodes = n_nodes
            self.n_steps = n_steps
            self.tolerance = tolerance
            self.activation = activation
            self.leak = leak

            self.register_buffer(
                'source_idx',
                torch.as_tensor(source_idx, dtype=torch.long, device=device),
            )
            self.register_buffer(
                'target_idx',
                torch.as_tensor(target_idx, dtype=torch.long, device=device),
            )
            self.register_buffer(
                'input_idx',
                torch.as_tensor(input_idx, dtype=torch.long, device=device),
            )
            self.register_buffer(
                'output_idx',
                torch.as_tensor(output_idx, dtype=torch.long, device=device),
            )

            edge_signs_tensor = torch.as_tensor(
                edge_signs,
                dtype=dtype,
                device=device,
            )
            known = torch.isin(
                edge_signs_tensor,
                torch.as_tensor([-1.0, 1.0], dtype=dtype, device=device),
            )

            initial_edges = torch.where(
                known,
                0.1 * edge_signs_tensor,
                torch.full_like(edge_signs_tensor, 0.1),
            )
            initial_edges = initial_edges + 0.01 * torch.randn_like(initial_edges)

            self.edge_weights = torch.nn.Parameter(initial_edges)
            self.bias = torch.nn.Parameter(
                torch.zeros(n_nodes, dtype=dtype, device=device),
            )
            self.output_scale = torch.nn.Parameter(
                torch.ones(len(output_idx), dtype=dtype, device=device),
            )
            self.output_bias = torch.nn.Parameter(
                torch.zeros(len(output_idx), dtype=dtype, device=device),
            )

            input_scale = torch.full(
                (len(input_idx),), input_scale_init, dtype=dtype, device=device,
            )

            if learn_input_scale:
                self.input_scale = torch.nn.Parameter(input_scale)
            else:
                self.register_buffer('input_scale', input_scale)

            self.register_buffer('edge_signs', edge_signs_tensor)
            self.register_buffer('known_signs', known)

        def edge_matrix(self):

            weights = torch.zeros(
                (self.n_nodes, self.n_nodes),
                dtype=self.edge_weights.dtype,
                device=self.edge_weights.device,
            )
            weights[self.target_idx, self.source_idx] = self.edge_weights

            return weights

        def _activate(self, values):

            if self.activation == 'mml':
                return _mml_activation(values, self.leak)

            if self.activation == 'tanh':
                return torch.tanh(values)

            if self.activation == 'sigmoid':
                return torch.sigmoid(values)

            if self.activation == 'leaky_relu':
                return torch.nn.functional.leaky_relu(
                    values,
                    negative_slope=self.leak,
                )

            raise ValueError(
                '`activation` must be one of mml, tanh, sigmoid or leaky_relu.'
            )

        def forward(self, x):

            drive = torch.zeros(
                (x.shape[0], self.n_nodes),
                dtype=x.dtype,
                device=x.device,
            )
            drive[:, self.input_idx] = x * self.input_scale

            state = torch.zeros_like(drive)
            weights = self.edge_matrix()

            for _ in range(self.n_steps):
                prev = state
                state = self._activate(state @ weights.T + drive + self.bias)

                if self.tolerance > 0:
                    if torch.max(torch.abs(state - prev)).item() < self.tolerance:
                        break

            prediction = (
                state[:, self.output_idx] * self.output_scale +
                self.output_bias
            )

            return prediction, state

        def uniform_regularization(
                self,
                state: 'torch.Tensor',
                target_min: float = 0.0,
                target_max: float = 1.0,
            ) -> 'torch.Tensor':
            """Penalize deviation of node states from a uniform distribution.

            Matches the LEMBAS uniform loss: pushes the distribution of node
            activations across samples to be roughly uniform in
            [target_min, target_max], which keeps states biologically
            interpretable and prevents saturation.
            """
            n = state.shape[0]
            sorted_state, _ = torch.sort(state, dim=0)
            target = torch.linspace(
                target_min, target_max, n,
                dtype=state.dtype, device=state.device,
            ).unsqueeze(1)

            dist_loss = torch.sum((sorted_state - target) ** 2)
            below = torch.sum(
                state.lt(target_min) * (state - target_min) ** 2
            )
            above = torch.sum(
                state.gt(target_max) * (state - target_max) ** 2
            )

            return dist_loss + below + above

        def sign_regularization(self):

            if not torch.any(self.known_signs):
                return torch.zeros((), dtype=self.edge_weights.dtype,
                                   device=self.edge_weights.device)

            signed_weights = self.edge_weights[self.known_signs]
            signs = self.edge_signs[self.known_signs]

            return torch.mean(torch.relu(-signed_weights * signs))

    return LembasRNN


def _edge_subnetwork(
        network: nx.DiGraph,
        edge_table: pd.DataFrame,
        min_abs_weight: float,
    ) -> nx.DiGraph:

    subnetwork = nx.DiGraph()

    for row in edge_table.itertuples(index=False):
        if abs(row.lembas_weight) < min_abs_weight:
            continue

        data = dict(network.get_edge_data(row.source, row.target, default={}))
        data['lembas_weight'] = row.lembas_weight
        data['lembas_abs_weight'] = abs(row.lembas_weight)
        subnetwork.add_edge(row.source, row.target, **data)

    return subnetwork


def run_lembas_rnn(
        network: nx.DiGraph,
        perturbations_train: pd.DataFrame,
        readouts_train: pd.DataFrame,
        perturbations_eval: pd.DataFrame | None = None,
        epochs: int = 1000,
        learning_rate: float = 2e-3,
        n_steps: int = 100,
        tolerance: float = 1e-5,
        alpha: float = 1e-6,
        sign_penalty: float = 0.1,
        uniform_penalty: float = 1e-4,
        batch_size: int | None = None,
        activation: str = 'mml',
        leak: float = 0.01,
        input_scale_init: float = 3.0,
        device: str = 'auto',
        dtype: str = 'float32',
        seed: int | None = 888,
        learn_input_scale: bool = False,
        min_abs_edge_weight: float = 0.0,
        verbose: bool = False,
    ) -> dict[str, t.Any]:
    """
    Train a LEMBAS-like recurrent model on perturbation-response data.

    Constrains recurrent edges to the supplied prior knowledge network,
    iterates node states until steady state (or ``n_steps`` max), and
    learns edge weights plus output projection parameters by minimizing
    readout MSE.

    Matches the core architecture of Nilsson et al. 2022 (Nat Commun):

    * Default MML (Michaelis-Menten-like) activation maps states to [0, 1].
    * Steady-state convergence check (``tolerance``) mirrors the original
      RNN stopping criterion.
    * Sign regularization penalizes edges whose learned sign contradicts
      the known mechanism of action.
    * Uniform regularization pushes node-state distributions toward uniform,
      keeping activations biologically interpretable.

    Use :func:`networkcommons.utils.lembas_format_network` to add the
    ``mode_of_action`` edge attribute before calling this function, and
    :func:`networkcommons.utils.network_from_df` to convert the edge table
    to a ``nx.DiGraph``.
    """

    if epochs <= 0:
        raise ValueError('`epochs` must be positive.')

    if learning_rate <= 0:
        raise ValueError('`learning_rate` must be positive.')

    if n_steps <= 0:
        raise ValueError('`n_steps` must be positive.')

    if tolerance < 0:
        raise ValueError('`tolerance` must be non-negative.')

    if alpha < 0 or sign_penalty < 0 or uniform_penalty < 0:
        raise ValueError('Regularization strengths must be non-negative.')

    if batch_size is not None and batch_size <= 0:
        raise ValueError('`batch_size` must be positive when provided.')

    if torch is None:
        raise ImportError(
            '`run_lembas_rnn` requires PyTorch. Install NetworkCommons with '
            'the `torch` extra: pip install networkcommons[torch]'
        )

    torch_dtype = _torch_dtype(dtype)
    torch_device = _torch_device(device)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    perturbations_train = _as_dataframe(
        perturbations_train,
        'perturbations_train',
    )
    readouts_train = _as_dataframe(readouts_train, 'readouts_train')
    perturbations_train, readouts_train = _align_samples(
        perturbations_train,
        readouts_train,
    )

    if perturbations_eval is None:
        perturbations_eval = perturbations_train
    else:
        perturbations_eval = _as_dataframe(
            perturbations_eval,
            'perturbations_eval',
        )
        perturbations_eval = perturbations_eval.loc[
            :,
            perturbations_train.columns,
        ]

    pkn = network_to_perturbation_table(network)
    nodes = sorted(
        set(pkn['source']) |
        set(pkn['target']) |
        set(perturbations_train.columns) |
        set(readouts_train.columns)
    )
    node_idx = {node: idx for idx, node in enumerate(nodes)}

    source_idx = pkn['source'].map(node_idx).to_numpy(dtype=int)
    target_idx = pkn['target'].map(node_idx).to_numpy(dtype=int)
    edge_signs = pkn['mode_of_action'].to_numpy(dtype=float)
    input_idx = np.array(
        [node_idx[node] for node in perturbations_train.columns],
        dtype=int,
    )
    output_idx = np.array(
        [node_idx[node] for node in readouts_train.columns],
        dtype=int,
    )

    model_class = _make_lembas_model()
    model = model_class(
        n_nodes=len(nodes),
        source_idx=source_idx,
        target_idx=target_idx,
        edge_signs=edge_signs,
        input_idx=input_idx,
        output_idx=output_idx,
        n_steps=n_steps,
        tolerance=tolerance,
        activation=activation,
        leak=leak,
        dtype=torch_dtype,
        device=torch_device,
        learn_input_scale=learn_input_scale,
        input_scale_init=input_scale_init,
    )

    x_train = torch.as_tensor(
        perturbations_train.to_numpy(dtype=float),
        dtype=torch_dtype,
        device=torch_device,
    )
    y_train = torch.as_tensor(
        readouts_train.to_numpy(dtype=float),
        dtype=torch_dtype,
        device=torch_device,
    )
    x_eval = torch.as_tensor(
        perturbations_eval.to_numpy(dtype=float),
        dtype=torch_dtype,
        device=torch_device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    n_samples = x_train.shape[0]
    batch_size = n_samples if batch_size is None else min(batch_size, n_samples)

    _log(f'LEMBAS-RNN: training on {n_samples} samples using {torch_device}.')

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n_samples, device=torch_device)
        epoch_losses = []

        for start in range(0, n_samples, batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            prediction, full_state = model(x_train[idx])
            fit_loss = torch.mean((prediction - y_train[idx]) ** 2)
            l2_loss = sum(torch.sum(param ** 2) for param in model.parameters())
            sign_loss = model.sign_regularization()
            uniform_loss = model.uniform_regularization(full_state)
            loss = (
                fit_loss
                + alpha * l2_loss
                + sign_penalty * sign_loss
                + uniform_penalty * uniform_loss
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(fit_loss.detach().cpu()))

        mean_epoch_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_epoch_loss)

        if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
            _log(f'LEMBAS-RNN: epoch {epoch + 1}; mse={mean_epoch_loss:.6g}')

    model.eval()

    with torch.no_grad():
        y_pred, states = model(x_eval)

    predictions = pd.DataFrame(
        y_pred.detach().cpu().numpy(),
        index=perturbations_eval.index,
        columns=readouts_train.columns,
    )
    states = pd.DataFrame(
        states.detach().cpu().numpy(),
        index=perturbations_eval.index,
        columns=nodes,
    )
    edge_table = pkn.copy()
    edge_table['lembas_weight'] = model.edge_weights.detach().cpu().numpy()
    edge_table['lembas_abs_weight'] = edge_table['lembas_weight'].abs()

    subnetwork = _edge_subnetwork(
        network,
        edge_table,
        min_abs_edge_weight,
    )

    return {
        'model': model,
        'predictions': predictions,
        'states': states,
        'loss_history': loss_history,
        'edge_weights': edge_table,
        'subnetwork': subnetwork,
        'nodes': nodes,
        'input_nodes': list(perturbations_train.columns),
        'output_nodes': list(readouts_train.columns),
        'device': str(torch_device),
    }
