from typing import Iterable, Set
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from scipy.sparse import dok_matrix, identity, kron
from networkx import MultiDiGraph
from project.task2_automata_conversions import regex_to_dfa, graph_to_nfa


class AdjacencyMatrixFA:
    alphabet: Set[Symbol]
    states: Set[State]
    size: int
    state_index: dict[State, int]
    start_states: Set[State]
    final_states: Set[State]
    decomposed_adj_matrix: dict[Symbol, dok_matrix]

    def __init__(self, fa: NondeterministicFiniteAutomaton):
        self.alphabet = fa.symbols
        self.states = fa.states
        self.start_states = fa.start_states
        self.final_states = fa.final_states
        self.size = len(self.states)
        self.state_index = {s: i for i, s in enumerate(self.states)}

        decomposed_adj_matrix = {
            symbol: dok_matrix((self.size, self.size), dtype=bool)
            for symbol in self.alphabet
        }

        for s_from, transitions in fa.to_dict().items():
            for symbol, targets in transitions.items():
                targets_iterable = (
                    targets if hasattr(targets, "__iter__") else [targets]
                )
                for s_to in targets_iterable:
                    decomposed_adj_matrix[symbol][
                        self.state_index[s_from], self.state_index[s_to]
                    ] = True

        self.decomposed_adj_matrix = decomposed_adj_matrix

    def start_states_indices(self) -> set[int]:
        return {self.state_index[s] for s in self.start_states}

    def final_states_indices(self) -> set[int]:
        return {self.state_index[s] for s in self.final_states}

    def accepts(self, word: Iterable[Symbol]) -> bool:
        current_states = self.start_states_indices()
        for symbol in word:
            if symbol not in self.decomposed_adj_matrix:
                return False
            new_states = set()
            matr = self.decomposed_adj_matrix[symbol]
            for s_from in current_states:
                row = matr.getrow(s_from)
                for s_to in row.indices:
                    new_states.add(s_to)
            current_states = new_states
        return bool(current_states & self.final_states_indices())

    def transitive_closure(self) -> dok_matrix:
        size = self.size
        reach_matr = dok_matrix((size, size), dtype=bool)
        for matr in self.decomposed_adj_matrix.values():
            reach_matr += matr

        reach_matr += identity(size, dtype=bool)

        for k in range(size):
            for i in range(size):
                if reach_matr[i, k]:
                    for j in range(size):
                        if reach_matr[k, j]:
                            reach_matr[i, j] = True
        return reach_matr

    def is_empty(self) -> bool:
        transitive_closure = self.transitive_closure()

        for s_from in self.start_states_indices():
            for s_to in self.final_states_indices():
                if transitive_closure[s_from, s_to]:
                    return False
        return True


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    new_automaton = AdjacencyMatrixFA.__new__(AdjacencyMatrixFA)

    new_automaton.alphabet = automaton1.alphabet & automaton2.alphabet
    states = set()
    state_index = {}
    for i, s1 in enumerate(automaton1.states):
        for j, s2 in enumerate(automaton2.states):
            state = State((s1.value, s2.value))
            states.add(state)
            state_index[state] = i * automaton2.size + j
    new_automaton.states = states
    new_automaton.state_index = state_index
    new_automaton.size = len(states)

    new_automaton.start_states = {
        (s1, s2) for s1 in automaton1.start_states for s2 in automaton2.start_states
    }
    new_automaton.final_states = {
        (s1, s2) for s1 in automaton1.final_states for s2 in automaton2.final_states
    }

    decomposed_adj_matrix = {}

    for symbol in new_automaton.alphabet:
        matr1 = automaton1.decomposed_adj_matrix[symbol]
        matr2 = automaton2.decomposed_adj_matrix[symbol]
        decomposed_adj_matrix[symbol] = kron(matr1, matr2).todok()

    new_automaton.decomposed_adj_matrix = decomposed_adj_matrix
    return new_automaton


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_adj_matr = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_adj_matr = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    new_adj_matr = intersect_automata(regex_adj_matr, graph_adj_matr)
    transitive_closure = new_adj_matr.transitive_closure()
    state_pairs = set()
    for s_from in new_adj_matr.start_states:
        for s_to in new_adj_matr.final_states:
            ind_from = new_adj_matr.state_index[s_from]
            ind_to = new_adj_matr.state_index[s_to]
            if transitive_closure[ind_from, ind_to]:
                _, graph_state_from = s_from
                _, graph_state_to = s_to
                state_pairs.add((graph_state_from.value, graph_state_to.value))
    return state_pairs
