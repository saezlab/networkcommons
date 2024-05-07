import pygraphviz as pgv
import networkx as nx

def visualize_graph_simple(network,
                           source_dict,
                           target_dict,
                           prog='dot', 
                           is_sign_consistent=True):
    """
    Visualize the graph using the provided network.

    Args:
        network (nx.Graph): The network to visualize.
        source_dict (dict): A dictionary containing the sources and sign of perturbation.
        target_dict (dict): A dictionary containing the targets and sign of measurements.
        prog (str, optional): The layout program to use. Defaults to 'dot'.
        is_sign_consistent (bool, optional): If True, only visualize sign consistent paths. Defaults to True.
    """

    A = nx.nx_agraph.to_agraph(network)
    A.graph_attr['ratio'] = '1.2'

    sources = list(source_dict.keys())
    target_dict_flat = {}
    for key, value in target_dict.items():
        for sub_key, sub_value in value.items():
            target_dict_flat[sub_key] = sub_value
    targets = set(target_dict_flat.keys())


    # Add an intermediary invisible node and edges for layout control

    for node in A.nodes():
        n = node.get_name()
        if n in sources:
            fillcolor = 'steelblue'
            if source_dict.get(n, 1) > 0 and is_sign_consistent:
                color = 'forestgreen'
            elif source_dict.get(n, 1) < 0 and is_sign_consistent:
                color = 'tomato3'
            else:
                color = 'steelblue'
            node.attr['shape'] = 'circle'
            node.attr['color'] = color
            node.attr['style'] = 'filled'
            node.attr['fillcolor'] = fillcolor
            node.attr['label'] = ''
            node.attr['penwidth'] = 3
        elif n in targets:
            fillcolor = 'mediumpurple1'
            if target_dict.get(n, 1) > 0 and is_sign_consistent:
                color = 'forestgreen'
            elif target_dict.get(n, 1) < 0 and is_sign_consistent:
                color = 'tomato3'
            else:
                color = 'mediumpurple1'
            node.attr['shape'] = 'circle'
            node.attr['color'] = color
            node.attr['style'] = 'filled'
            node.attr['fillcolor'] = fillcolor
            node.attr['label'] = ''
            node.attr['penwidth'] = 3
        else:
            node.attr['shape'] = 'circle'
            node.attr['color'] = 'gray'
            node.attr['style'] = 'filled'
            node.attr['fillcolor'] = 'gray'
            node.attr['label'] = ''
    
    for edge in A.edges():
        u, v = edge
        edge_data = network.get_edge_data(u, v)
        if 'interaction' in edge_data:
            edge_data['sign'] = edge_data.pop('interaction')
        if edge_data['sign'] == 1 and is_sign_consistent:
            edge_color = 'forestgreen'
        elif edge_data['sign'] == -1 and is_sign_consistent:
            edge_color = 'tomato3'
        else:
            edge_color = 'gray30'
        edge.attr['color'] = edge_color
        edge.attr['penwidth'] = 2
    
    A.layout(prog=prog)
    return(A)
