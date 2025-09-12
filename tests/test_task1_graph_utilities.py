import pytest
from project.task1_graph_utilities import (
    get_graph_info_by_name,
    build_graph_from_two_cycles,
    save_graph_from_two_cycles_to_dot_file,
)
from utilities import get_graph_info_from_json, GRAPH_PARAMS
from networkx.drawing.nx_pydot import read_dot
from networkx.algorithms.isomorphism import is_isomorphic, categorical_edge_match


@pytest.mark.parametrize("graph_name, path_to_graph", GRAPH_PARAMS)
def test_get_graph_info_by_existing_name(graph_name, path_to_graph):
    actual_info = get_graph_info_by_name(graph_name)
    expected_info = get_graph_info_from_json(path_to_graph)
    assert actual_info == expected_info


# as stated in the source code of "download" function, which is used in "get_graph_info_by_name",
# if no graph with such name exists in the dataset (https://formallanguageconstrainedpathquerying.github.io/CFPQ_Data/graphs/index.html),
# a FileNotFoundError is raised
def test_get_graph_info_by_nonexisting_name():
    with pytest.raises(FileNotFoundError):
        get_graph_info_by_name("non-existing name")


@pytest.mark.parametrize(
    "node_num1, node_num2, labels",
    [(1, 1, ("hi", ",")), (4, 5, ("it", "is")), (100, 200, ("tea", "time"))],
)
def test_dot_file_contains_two_cycles_graph(node_num1, node_num2, labels, tmp_path):
    path = tmp_path / "graph.dot"
    expected_graph = build_graph_from_two_cycles(node_num1, node_num2, labels)
    save_graph_from_two_cycles_to_dot_file(node_num1, node_num2, labels, path)
    actual_graph = read_dot(path)
    edge_matcher = categorical_edge_match(
        "label", default=None
    )  # to compare labels of two graphs
    assert is_isomorphic(expected_graph, actual_graph, edge_match=edge_matcher)


@pytest.mark.parametrize(
    "node_num1, node_num2, labels",
    [(0, 0, ("have", "a")), (4, 0, ("great", "night")), (0, 1, ("or", "day"))],
)
def test_build_graph_with_out_of_range_indices_in_cycles(
    node_num1, node_num2, labels, tmp_path
):
    path = tmp_path / "graph.dot"
    with pytest.raises(IndexError):
        save_graph_from_two_cycles_to_dot_file(node_num1, node_num2, labels, path)
