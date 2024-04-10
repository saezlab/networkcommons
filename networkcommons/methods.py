import networkx as nx
from networkcommons.utils import get_subnetwork

def shortest_paths(network, source, target, verbose = False):
        """
        Calculate the shortest paths between sources and targets.

        Args:
            verbose (bool): If True, print warnings when no path is found to a given target.

        Returns:
            list: A list containing the shortest paths.
         """
        shortest_paths_res = []
        connected_targets = {}

        if type(source) == str:
            source = [source]
        elif type(source) == list:
            source = source
        elif type(source) == dict:
            source = list(source.keys())
        
        if type(target) == str:
            target = [target]
        elif type(target) == list:
            target = target
        elif type(target) == dict:
            target = list(target.keys())

        for source_node in source:
            if source_node not in connected_targets:
                connected_targets[source_node] = []

            for target_node in target:
                try:
                    shortest_paths_res.extend([p for p in nx.all_shortest_paths(network, source=source_node, target=target_node, weight='weight')])
                    connected_targets[source_node].append(target_node)
                except nx.NetworkXNoPath as e:
                    if verbose:
                        print(f"Warning: {e}")
                except nx.NodeNotFound as e:
                    if verbose:
                        print(f"Warning: {e}")

        subnetwork, connected_targets = get_subnetwork(network, shortest_paths_res)

        return subnetwork, shortest_paths_res

