from typing import Set
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from pyformlang.regular_expression import Regex
from networkx import MultiDiGraph


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    epsinlon_nfa = Regex(regex).to_epsilon_nfa()
    return epsinlon_nfa.minimize()


def nodes_to_states(graph: MultiDiGraph) -> dict[int, State]:
    return {int(node): State(int(node)) for node in graph.nodes}


def graph_to_nfa(
    graph: MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    state_map = nodes_to_states(graph)
    states = set(state_map.values())
    if not start_states:
        start_states = states
    if not final_states:
        final_states = states

    nfa = NondeterministicFiniteAutomaton(
        states=states, start_state=start_states, final_states=final_states
    )

    for v1, v2, label in graph.edges(data="label"):
        s_from, s_to, symbol = int(v1), int(v2), label
        if label is None:
            symbol = "$"
        nfa.add_transition(state_map[s_from], Symbol(symbol), state_map[s_to])
    return nfa
