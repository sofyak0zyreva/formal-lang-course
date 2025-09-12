import random
from networkx import MultiDiGraph
import pytest
import string
from pyformlang.regular_expression import MisformedRegexError
from project.task2_automata_conversions import regex_to_dfa, graph_to_nfa
from project.task1_graph_utilities import (
    get_graph_by_name,
    save_graph_from_two_cycles_to_dot_file,
    get_graph_info_from_graph,
)
from networkx.drawing.nx_pydot import read_dot
from utilities import GRAPH_PARAMS


class TestRegexToDfa:
    def test_empty_regex(self):
        dfa = regex_to_dfa("")
        assert dfa.is_empty()

    def test_only_accepts_epsilon(self):
        dfa = regex_to_dfa("$")
        assert dfa.accepts("")
        word = random.choice(string.punctuation)
        assert not dfa.accepts(word)

    def test_misformed_regex(self):
        with pytest.raises(MisformedRegexError):
            regex_to_dfa(" ( * ")

    def test_dfa_equivalence(self):
        dfa1 = regex_to_dfa("(a|b)*")
        dfa2 = regex_to_dfa("(a*b*)*")
        assert dfa1.is_equivalent_to(dfa2)


class TestGraphToNfa:
    def test_empty_graph(self):
        graph = MultiDiGraph()
        nfa = graph_to_nfa(graph, set(), set())
        assert nfa.is_empty()

    def test_graph_with_one_node(self):
        graph = MultiDiGraph()
        graph.add_node(0)
        # if the node is start+final, then nfa should accept epsilon
        nfa = graph_to_nfa(graph, set(), set())
        assert nfa.accepts("")
        # if the node is not start+final, then nfa should be empty
        nfa = graph_to_nfa(graph, {1}, {2})
        word = random.choice(string.punctuation)
        assert not nfa.accepts(word)

    @pytest.mark.parametrize(
        "node_num1, node_num2, labels",
        [(1, 1, ("hi", ",")), (4, 5, ("it", "is")), (100, 200, ("tea", "time"))],
    )
    def test_graph_from_dot(self, node_num1, node_num2, labels, tmp_path):
        path = tmp_path / "graph.dot"
        save_graph_from_two_cycles_to_dot_file(node_num1, node_num2, labels, path)
        graph = read_dot(path)
        nfa = graph_to_nfa(graph, set(), set())
        assert nfa.start_states == nfa.final_states == nfa.states
        assert len(nfa.states) == node_num1 + node_num2 + 1
        graph_info = get_graph_info_from_graph(graph)
        assert nfa.symbols == set(graph_info.edge_labels)
        assert nfa.get_number_transitions() == graph_info.edges_num

    @pytest.mark.parametrize("graph_name, path_to_graph", GRAPH_PARAMS)
    def test_graph_from_dataset(self, graph_name, path_to_graph):
        graph = get_graph_by_name(graph_name)
        graph_info = get_graph_info_from_graph(graph)
        nfa = graph_to_nfa(graph, set(), set())
        assert nfa.start_states == nfa.final_states == nfa.states
        assert len(nfa.states) == graph_info.nodes_num
        assert nfa.get_number_transitions() == graph_info.edges_num

    def test_graph_distinct_final_and_start_states(self):
        graph = MultiDiGraph()
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edges_from([(0, 1), (1, 3), (3, 2)])
        nfa = graph_to_nfa(graph, {0, 1}, {2, 3})
        # no node is start+final, so nfa should not accept epsilon
        assert not nfa.accepts("")

    def test_disconnected_graph(self):
        graph = MultiDiGraph()
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edges_from([(0, 1), (1, 3), (3, 0)])
        nfa = graph_to_nfa(
            graph, {1}, {2}
        )  # start state in one component, final state in another
        assert nfa.is_empty()

    def test_parallel_edges(self):
        graph = MultiDiGraph()
        graph.add_edge(0, 1, label="a")
        graph.add_edge(0, 1, label="b")
        nfa = graph_to_nfa(graph, {0}, {1})
        # nfa should accept both "a" and "b"
        assert nfa.accepts("a")
        assert nfa.accepts("b")
