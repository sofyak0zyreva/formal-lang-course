from dataclasses import dataclass
from typing import List, Tuple
import cfpq_data
from networkx.drawing.nx_pydot import write_dot
import networkx as nx


@dataclass
class GraphInfo:
    nodes_num: int
    edges_num: int
    edge_labels: List[str]


def get_graph_by_name(name: str) -> nx.MultiDiGraph:
    path = cfpq_data.download(name)
    return cfpq_data.graph_from_csv(path)


def get_graph_info_by_name(name: str) -> GraphInfo:
    graph = get_graph_by_name(name)
    return GraphInfo(
        nodes_num=graph.number_of_nodes(),
        edges_num=graph.number_of_edges(),
        edge_labels=cfpq_data.get_sorted_labels(graph),
    )


# a helper function useful for tests
def get_graph_info_from_graph(graph: nx.MultiDiGraph):
    return GraphInfo(
        nodes_num=graph.number_of_nodes(),
        edges_num=graph.number_of_edges(),
        edge_labels=cfpq_data.get_sorted_labels(graph),
    )


def save_graph_from_two_cycles_to_dot_file(
    node_num_1: int, node_num_2: int, labels: Tuple[str, str], path: str
):
    graph = cfpq_data.labeled_two_cycles_graph(node_num_1, node_num_2, labels=(labels))
    write_dot(graph, path)


# this function is just a wrapping of the library function, so there's hardly any need to test it
def build_graph_from_two_cycles(
    node_num_1: int, node_num_2: int, labels: Tuple[str, str]
) -> nx.MultiDiGraph:
    return cfpq_data.labeled_two_cycles_graph(node_num_1, node_num_2, labels=(labels))
