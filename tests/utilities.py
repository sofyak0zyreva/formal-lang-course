import json
from pathlib import Path
from project.task1_graph_utilities import GraphInfo


DATA_DIR = Path(__file__).parent / "resources/graph_datasets"

GRAPH_NAMES = ["biomedical", "pathways", "wine"]

GRAPH_PARAMS = [(name, DATA_DIR / f"{name}_graph_info.json") for name in GRAPH_NAMES]


def get_graph_info_from_json(path: str) -> GraphInfo:
    with open(path) as f:
        data = json.load(f)

    return GraphInfo(
        nodes_num=data["nodes_num"],
        edges_num=data["edges_num"],
        edge_labels=data["edge_labels"],
    )
