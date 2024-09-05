import networkx as nx
import random
import csv


def generate_network(num_nodes, num_sources=3, num_targets=3, filename="network.csv"):
    """
    Generates a directed network and saves it to a CSV file.

    Args:
        num_nodes (int): The number of nodes in the network.
        num_sources (int): The number of source nodes.
        num_targets (int): The number of target nodes.
        filename (str): The name of the file to save the network.
    """
    G = nx.DiGraph()

    # Create nodes
    nodes = [f"Node{i}" for i in range(num_nodes)]
    G.add_nodes_from(nodes)

    # Randomly choose sources and targets
    sources = random.sample(nodes, num_sources)
    targets = random.sample(nodes, num_targets)

    # Mark sources and targets
    for node in sources:
        G.nodes[node]['type'] = 'source'

    for node in targets:
        G.nodes[node]['type'] = 'target'

    # Create edges with random effects
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < 0.1:  # 10% chance of creating an edge
                effect = random.choice(['1', '-1'])
                G.add_edge(nodes[i], nodes[j], effect=effect)

    # Ensure the network is sparse but fully connected
    for target in targets:
        if not any(G.predecessors(target)):
            source = random.choice(sources)
            effect = random.choice(['1', '-1'])
            G.add_edge(source, target, effect=effect)

    # Save to CSV
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Node1", "Effect", "Node2", "Type"])
        for u, v, data in G.edges(data=True):
            node1_type = G.nodes[u].get('type', 'none')
            node2_type = G.nodes[v].get('type', 'none')
            writer.writerow([u, data['effect'], v, node1_type if node1_type != 'none' else node2_type])


# Generate networks of different sizes
generate_network(10, filename="small_network.csv")
generate_network(20, filename="medium_network.csv")
generate_network(100, filename="large_network.csv")
generate_network(1000, filename="extra_large_network.csv")