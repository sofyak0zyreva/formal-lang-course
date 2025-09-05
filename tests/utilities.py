import json
from project.task1_graph_utilities import GraphInfo


def get_graph_info_from_json(path: str) -> GraphInfo:
    with open(path) as f:
        data = json.load(f)

    return GraphInfo(
        nodes_num=data["nodes_num"],
        edges_num=data["edges_num"],
        edge_labels=data["edge_labels"],
    )
