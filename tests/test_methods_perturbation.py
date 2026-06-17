import networkx as nx
import numpy as np
import pandas as pd
import pytest

from networkcommons.methods import _perturbation


def _toy_network():

    network = nx.DiGraph()
    network.add_edge('Ligand', 'Kinase', sign=1)
    network.add_edge('Kinase', 'TF1', sign=-1)
    network.add_edge('Ligand', 'TF2', weight=2.0)
    network.add_edge('Unknown', 'TF2')

    return network


def test_network_to_perturbation_table_infers_signs():

    table = _perturbation.network_to_perturbation_table(_toy_network())

    assert table.to_dict('records') == [
        {'source': 'Ligand', 'target': 'Kinase', 'mode_of_action': 1.0},
        {'source': 'Ligand', 'target': 'TF2', 'mode_of_action': 1.0},
        {'source': 'Kinase', 'target': 'TF1', 'mode_of_action': -1.0},
        {'source': 'Unknown', 'target': 'TF2', 'mode_of_action': 0.1},
    ]


def test_split_perturbation_data_aligns_and_splits_samples():

    perturbations = pd.DataFrame(
        {'Ligand': [0, 1, 0, 1]},
        index=['s1', 's2', 's3', 's4'],
    )
    readouts = pd.DataFrame(
        {'TF1': [0.1, 0.5, 0.2, 0.6]},
        index=['s4', 's3', 's2', 's1'],
    )

    x_train, x_test, y_train, y_test = _perturbation.split_perturbation_data(
        perturbations,
        readouts,
        test_size=1,
        seed=1,
    )

    assert len(x_train) == 3
    assert len(x_test) == 1
    assert list(x_train.index) == list(y_train.index)
    assert list(x_test.index) == list(y_test.index)


def test_mean_response_baseline_predicts_training_mean():

    readouts_train = pd.DataFrame(
        {'TF1': [1.0, 3.0], 'TF2': [2.0, 6.0]},
        index=['s1', 's2'],
    )
    readouts_eval = pd.DataFrame(index=['s3', 's4'])

    result = _perturbation.run_mean_response_baseline(
        readouts_train,
        readouts_eval,
    )

    expected = pd.DataFrame(
        {'TF1': [2.0, 2.0], 'TF2': [4.0, 4.0]},
        index=['s3', 's4'],
    )

    pd.testing.assert_frame_equal(result['predictions'], expected)


def test_ridge_baseline_fits_linear_response():

    perturbations = pd.DataFrame(
        {
            'LigandA': [0, 1, 2, 3, 4],
            'LigandB': [1, 0, 1, 0, 1],
        },
        index=[f's{i}' for i in range(5)],
    )
    readouts = pd.DataFrame(
        {
            'TF1': 1 + 2 * perturbations['LigandA'] - perturbations['LigandB'],
            'TF2': -2 + perturbations['LigandA'],
        },
        index=perturbations.index,
    )

    result = _perturbation.run_ridge_baseline(
        perturbations,
        readouts,
        alpha=1e-8,
    )
    metrics = _perturbation.evaluate_predictions(
        readouts,
        result['predictions'],
    )

    assert result['train_mse'] < 1e-12
    assert metrics.loc['__all__', 'mse'] < 1e-12
    assert set(result['coefficients'].index) == {'LigandA', 'LigandB'}


def test_evaluate_predictions_axis_readout():

    y_true = pd.DataFrame(
        {'TF1': [0.0, 1.0], 'TF2': [2.0, 3.0]},
        index=['s1', 's2'],
    )
    y_pred = pd.DataFrame(
        {'TF1': [0.5, 1.5], 'TF2': [2.0, 3.0]},
        index=['s1', 's2'],
    )

    metrics = _perturbation.evaluate_predictions(y_true, y_pred, axis='readout')

    assert metrics.index.name == 'readout'
    assert set(metrics.index) == {'TF1', 'TF2', '__all__'}
    assert metrics.loc['TF2', 'mse'] == pytest.approx(0.0)
    assert metrics.loc['TF1', 'mse'] == pytest.approx(0.25)


def test_evaluate_predictions_axis_condition():

    y_true = pd.DataFrame(
        {'TF1': [0.0, 1.0], 'TF2': [2.0, 3.0]},
        index=['s1', 's2'],
    )
    y_pred = pd.DataFrame(
        {'TF1': [0.0, 1.0], 'TF2': [2.5, 3.0]},
        index=['s1', 's2'],
    )

    metrics = _perturbation.evaluate_predictions(y_true, y_pred, axis='condition')

    assert metrics.index.name == 'condition'
    assert set(metrics.index) == {'s1', 's2', '__all__'}
    assert metrics.loc['s2', 'mse'] == pytest.approx(0.0)
    assert metrics.loc['s1', 'mse'] == pytest.approx(0.25 / 2)  # one TF off by 0.5


def test_evaluate_predictions_bad_axis():

    y = pd.DataFrame({'TF1': [1.0]}, index=['s1'])

    with pytest.raises(ValueError, match='axis'):
        _perturbation.evaluate_predictions(y, y, axis='feature')


def test_lembas_rnn_smoke():

    pytest.importorskip('torch')

    perturbations = pd.DataFrame(
        {'Ligand': [0.0, 0.5, 1.0, 1.5]},
        index=['s1', 's2', 's3', 's4'],
    )
    readouts = pd.DataFrame(
        {
            'TF1': [-0.1, -0.3, -0.5, -0.7],
            'TF2': [0.1, 0.2, 0.3, 0.4],
        },
        index=perturbations.index,
    )

    result = _perturbation.run_lembas_rnn(
        _toy_network(),
        perturbations,
        readouts,
        epochs=2,
        n_steps=2,
        learning_rate=1e-2,
        seed=1,
    )

    assert list(result['predictions'].index) == list(perturbations.index)
    assert list(result['predictions'].columns) == ['TF1', 'TF2']
    assert len(result['loss_history']) == 2
    assert result['subnetwork'].number_of_edges() == _toy_network().number_of_edges()
