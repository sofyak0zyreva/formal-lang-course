from dataclasses import dataclass
from typing import Any, List, Tuple
import cfpq_data
from networkx.drawing.nx_pydot import write_dot


@dataclass
class GraphInfo:
    nodes_num: int
    edges_num: int
    edge_labels: List[Any]


def get_graph_info_by_name(name: str) -> GraphInfo:
    path = cfpq_data.download(name)
    graph = cfpq_data.graph_from_csv(path)
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


# def save_built_graph_to_dot_file(graph: nx.MultiDiGraph, path: str):
#     write_dot(graph, path)

# g = get_graph_info_by_name("wine")
# print(g.edge_labels)
save_graph_from_two_cycles_to_dot_file(4, 5, ("a", "b"), "my.dot")
