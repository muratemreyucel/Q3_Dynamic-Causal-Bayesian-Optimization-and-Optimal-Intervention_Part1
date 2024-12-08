from itertools import chain, combinations
import networkx as nx
import pygraphviz as pgv
from networkx.drawing.nx_agraph import write_dot, read_dot
import matplotlib.pyplot as plt


def powerset(iterable):
    """
    Generate all subsets of a given iterable.
    """
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


def generate_causal_graph(T=3):
    """
    Generate a causal graph with 15 nodes at each time step.
    Includes 7 manipulable nodes, 7 non-manipulable nodes, and 1 target variable.

    Parameters
    ----------
    T : int
        Number of time slices.

    Returns
    -------
    nx.DiGraph
        A directed causal graph.
    """
    if T < 1:
        raise ValueError("T must be greater than or equal to 1.")

    G = nx.DiGraph()

    # Define nodes
    manipulable_nodes = [f"M{i}" for i in range(1, 8)]
    non_manipulable_nodes = [f"N{i}" for i in range(1, 8)]
    target_node = ["T"]

    all_nodes = manipulable_nodes + non_manipulable_nodes + target_node

    # Add temporal and causal dependencies
    for t in range(T):
        # Add nodes for each time slice
        for node in all_nodes:
            G.add_node(f"{node}_{t}")

        # Add temporal dependencies
        if t > 0:
            for node in all_nodes:
                G.add_edge(f"{node}_{t-1}", f"{node}_{t}")

        # Add topological dependencies (example: N depends on M)
        for m_node in manipulable_nodes:
            for n_node in non_manipulable_nodes:
                G.add_edge(f"{m_node}_{t}", f"{n_node}_{t}")

        # Add dependencies on the target node
        for n_node in non_manipulable_nodes:
            G.add_edge(f"{n_node}_{t}", f"T_{t}")

    return G


def get_exploration_set(manipulable_nodes, max_simultaneous_interventions=3):
    """
    Generate the exploration set (subsets of manipulable nodes for interventions).

    Parameters
    ----------
    manipulable_nodes : list
        List of manipulable node names.
    max_simultaneous_interventions : int
        Maximum number of simultaneous interventions.

    Returns
    -------
    list
        List of subsets of manipulable nodes.
    """
    exploration_sets = [
        s for s in powerset(manipulable_nodes) if 0 < len(s) <= max_simultaneous_interventions
    ]
    exploration_sets.append(())  # Add empty set for no intervention
    return exploration_sets


def save_graph_as_png(graph, filename="causal_graph.png"):
    """
    Save the causal graph as a PNG file.

    Parameters
    ----------
    graph : nx.DiGraph
        The causal graph to save.
    filename : str
        The file name for the saved image.
    """
    agraph = nx.nx_agraph.to_agraph(graph)
    agraph.draw(filename, format="png", prog="dot")
    print(f"Graph saved as {filename}")


def save_graph_to_dot(graph, filename="causal_graph.dot"):
    """
    Save the causal graph to a DOT file.

    Parameters
    ----------
    graph : nx.DiGraph
        The causal graph to save.
    filename : str
        The file name for the saved DOT file.
    """
    write_dot(graph, filename)
    print(f"Graph saved as {filename}")


def visualize_graph_from_dot(filename="causal_graph.dot"):
    """
    Load and visualize the causal graph from a DOT file.

    Parameters
    ----------
    filename : str
        The DOT file to load and visualize.
    """
    graph = read_dot(filename)
    plt.figure(figsize=(12, 8))
    nx.draw(
        graph,
        with_labels=True,
        node_color=[
            "lightblue" if "M" in node else "lightgreen" if "N" in node else "orange"
            for node in graph.nodes()
        ],
        node_size=500,
        font_size=8,
        edge_color="gray",
    )
    plt.title("Causal Graph Visualization")
    plt.show()


if __name__ == "__main__":
    # Number of time slices
    T = 3

    # Generate causal graph
    causal_graph = generate_causal_graph(T)

    # Save the graph as PNG
    save_graph_as_png(causal_graph, "causal_graph.png")

    # Save the graph to a DOT file
    save_graph_to_dot(causal_graph, "causal_graph.dot")

    # Define manipulable nodes
    manipulable_nodes = [f"M{i}" for i in range(1, 8)]

    # Get exploration set
    exploration_sets = get_exploration_set(manipulable_nodes, max_simultaneous_interventions=3)
    print("Exploration Sets:", exploration_sets)

    # Visualize the graph
    visualize_graph_from_dot("causal_graph.dot")
