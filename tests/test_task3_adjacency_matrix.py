from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from project.task3_adjacency_matrix import (
    AdjacencyMatrixFA,
    intersect_automata,
    tensor_based_rpq,
)
from project.task2_automata_conversions import regex_to_dfa
from networkx import MultiDiGraph


def test_accepts_epsilon_only():
    nfa = NondeterministicFiniteAutomaton()
    s = State(0)
    nfa.add_start_state(s)
    nfa.add_final_state(s)
    am = AdjacencyMatrixFA(nfa)
    assert am.accepts("")
    assert not am.accepts("a")


def test_empty_automaton():
    nfa = NondeterministicFiniteAutomaton()
    am = AdjacencyMatrixFA(nfa)
    assert am.is_empty()


def test_is_empty():
    # empty language: unreachable final state
    nfa2 = NondeterministicFiniteAutomaton()
    nfa2.add_transition(State(0), Symbol("a"), State(1))
    nfa2.add_start_state(State(0))
    nfa2.add_final_state(State(2))
    amfa2 = AdjacencyMatrixFA(nfa2)
    assert amfa2.is_empty()

    # path 0 -> 1 -> 2
    nfa3 = NondeterministicFiniteAutomaton()
    nfa3.add_transition(State(0), Symbol("a"), State(1))
    nfa3.add_transition(State(1), Symbol("a"), State(2))
    nfa3.add_start_state(State(0))
    nfa3.add_final_state(State(2))
    amfa3 = AdjacencyMatrixFA(nfa3)
    assert not amfa3.is_empty()


def test_intersect_simple():
    dfa1 = AdjacencyMatrixFA(regex_to_dfa("a"))
    dfa2 = AdjacencyMatrixFA(regex_to_dfa("a*"))
    inter = intersect_automata(dfa1, dfa2)
    assert inter.accepts("a")
    assert not inter.accepts("aa")


def test_tensor_rpq_simple_graph():
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    result = tensor_based_rpq("a", graph, {0}, {1})
    assert (0, 1) in result


def test_tensor_rpq_multiple_start_and_final_states():
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    result = tensor_based_rpq("a b", graph, {0, 1}, {1, 2})
    assert (0, 2) in result


def test_tensor_rpq_unreachable():
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="b")
    result = tensor_based_rpq("a", graph, {0}, {1})
    assert result == set()
