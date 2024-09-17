import networkx as nx
import corneto as cn
from networkcommons.methods import _causal
from unittest.mock import patch, MagicMock
import io
import sys
import pytest

@patch('networkcommons.methods._causal._log')  # Mock the _log function
def test_run_corneto_carnival(mock_log):
    # From CORNETO docs:
    network = cn.Graph.from_sif_tuples([
        ('I1', 1, 'N1'),  # I1 activates N1
        ('N1', 1, 'M1'),  # N1 activates M1
        ('N1', 1, 'M2'),  # N1 activates M2
        ('I2', -1, 'N2'),  # I2 inhibits N2
        ('N2', -1, 'M2'),  # N2 inhibits M2
        ('N2', -1, 'M1'),  # N2 inhibits M1
    ])

    source_dict = {'I1': 1}
    target_dict = {'M1': 1, 'M2': 1}

    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    sys.stdout = stdout_capture  # Redirect stdout
    sys.stderr = stderr_capture  # Redirect stderr

    try:
        # Run the modified function with verbose=False
        result_network = _causal.run_corneto_carnival(
            network,
            source_dict,
            target_dict,
            betaWeight=0.1,
            solver='scipy',
            verbose=False  # We test without verbosity first
        )
    finally:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    # Check that the result is as expected
    test_network = nx.DiGraph()
    test_network.add_edge('I1', 'N1')
    test_network.add_edge('N1', 'M1')
    test_network.add_edge('N1', 'M2')

    # Check if the resulting network is isomorphic to the expected test network
    assert nx.is_isomorphic(result_network, test_network)
    assert isinstance(result_network, nx.Graph)
    assert list(result_network.nodes) == ['I1', 'N1', 'M1', 'M2']
    assert '_s' not in result_network.nodes
    assert '_pert_c0' not in result_network.nodes
    assert '_meas_c0' not in result_network.nodes

    # Ensure the log was called with the correct messages
    mock_log.assert_any_call('Running Vanilla Carnival algorithm via CORNETO...')
    mock_log.assert_any_call('CORNETO-Carnival finished.')
    mock_log.assert_any_call(f'Network solution with {len(result_network.nodes)} nodes and {len(result_network.edges)} edges.')

    # Optionally check captured stdout and stderr if needed
    stdout_output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()

    assert stdout_output == ""  # Since verbose=False, no output should be printed to stdout
    assert stderr_output == ""  # Similarly, no stderr output expected in this case

    # Now test with verbose=True
    sys.stdout = io.StringIO()  # Redirect stdout again to capture print statements

    try:
        result_network_verbose = _causal.run_corneto_carnival(
            network,
            source_dict,
            target_dict,
            betaWeight=0.1,
            solver='scipy',
            verbose=True  # Verbose=True, so output should be printed
        )
    finally:
        captured_verbose_output = sys.stdout.getvalue()
        sys.stdout = old_stdout  # Restore stdout
    
    # testing some lines of the stdout as a proxy for the whole log
    lines_in_output = captured_verbose_output.splitlines()
    assert any("CVXPY" in line for line in lines_in_output)
    assert any("Solver terminated with message: Optimization terminated successfully. (HiGHS Status 7: Optimal)" in line for line in lines_in_output)
    # Check that stdout has printed lines when verbose=True
    assert len(captured_verbose_output) > 0

    # TODO: why does corneto raise a TypeError? Thought it was the empty network, but it's not
    # with pytest.raises(TypeError):
    #     _causal.run_corneto_carnival(
    #         nx.DiGraph(),  # Empty network
    #         source_dict,
    #         target_dict,
    #         betaWeight=0.1,
    #         solver='scipy',
    #         verbose=True
    #     )
