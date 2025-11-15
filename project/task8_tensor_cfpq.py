import pyformlang
import networkx as nx
import pyformlang.rsa
from pyformlang.finite_automaton import (
    State,
    NondeterministicFiniteAutomaton,
    Symbol,
)
from scipy.sparse import dok_matrix


from project.task2_automata_conversions import graph_to_nfa
from project.task3_adjacency_matrix import AdjacencyMatrixFA, intersect_automata


def rsm_state(sym: Symbol, state: State) -> State:
    return State((sym, state))


def rsm_to_adj_matr(
    rsm: pyformlang.rsa.RecursiveAutomaton,
) -> pyformlang.finite_automaton.NondeterministicFiniteAutomaton:
    start_states = set()
    final_states = set()
    transitions = []

    for nonterm, box in rsm.boxes.items():
        for s_from, s_to, lbl in box.dfa.to_networkx().edges(data="label"):
            transitions.append(
                (rsm_state(nonterm, s_from), lbl, rsm_state(nonterm, s_to))
            )

        for start_state in box.start_state:
            new_start = rsm_state(nonterm, start_state)
            start_states.add(new_start)

        for final_state in box.final_states:
            new_final = rsm_state(nonterm, final_state)
            final_states.add(new_final)

    rsm_nfa = NondeterministicFiniteAutomaton(
        start_state=start_states, final_states=final_states
    )
    rsm_nfa.add_transitions(transitions)
    return AdjacencyMatrixFA(rsm_nfa)


def ms_bfs(rsm_am: AdjacencyMatrixFA, start_rsm: set[State], am: AdjacencyMatrixFA):
    m = len(start_rsm)
    n = am.size

    front_matrix = dok_matrix((m, n), dtype=bool)
    for i, st in enumerate(start_rsm):
        front_matrix[i, am.state_index[st]] = True

    visited = front_matrix
    while front_matrix.count_nonzero() > 0:
        new_front = {}
        for symbol in am.alphabet:
            new_front[symbol] = front_matrix @ am.decomposed_adj_matrix[symbol]
        front_matrix = sum(new_front.values()) > visited
        visited = visited + front_matrix

    graph_transitions = set()
    for i, s_from in enumerate(start_rsm):
        for s_to in am.states:
            if visited[i, am.state_index[s_to]]:
                s_rsm_from, s_graph_from = s_from.value
                nonterm, st = s_rsm_from
                s_rsm_to, s_graph_to = s_to.value
                if (
                    s_rsm_from in rsm_am.start_states
                    and s_rsm_to in rsm_am.final_states
                ):
                    graph_transitions.add((nonterm, s_graph_from, s_graph_to))
    return graph_transitions


def index_to_state(m: AdjacencyMatrixFA) -> dict[int, State]:
    res = {}
    for state, idx in m.state_index.items():
        res[idx] = state
    return res


def tensor_based_cfpq(
    rsm: pyformlang.rsa.RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    rsm_am = rsm_to_adj_matr(rsm)
    graph_am = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    while True:
        am = intersect_automata(rsm_am, graph_am)
        start_rsm: set[State] = set()
        # choose intersection states where rsm state is start state
        for state in am.states:
            st_rsm, _ = state.value
            if st_rsm in rsm_am.start_states:
                start_rsm.add(state)
        transitions = ms_bfs(rsm_am, start_rsm, am)
        new_added = 0
        # adding nonterminals to graph adj matr
        for sym, s_from, s_to in transitions:
            if sym not in graph_am.decomposed_adj_matrix.keys():
                graph_am.decomposed_adj_matrix[sym] = dok_matrix(
                    (graph_am.size, graph_am.size), dtype=bool
                )
                (graph_am.alphabet).add(sym)
            i = graph_am.state_index[s_from]
            j = graph_am.state_index[s_to]
            if not graph_am.decomposed_adj_matrix[sym].get((i, j), False):
                graph_am.decomposed_adj_matrix[sym][i, j] = True
                new_added += 1
        if new_added == 0:
            break

    result = set()
    if rsm.initial_label in graph_am.decomposed_adj_matrix.keys():
        matr = graph_am.decomposed_adj_matrix[rsm.initial_label]
        rows, cols = matr.nonzero()
        for i, j in zip(rows, cols):
            st = index_to_state(graph_am)
            s = st[i]
            f = st[j]
            if (s in graph_am.start_states) and (f in graph_am.final_states):
                result.add((s, f))
    return result


def cfg_to_rsm(cfg: pyformlang.cfg.CFG) -> pyformlang.rsa.RecursiveAutomaton:
    return pyformlang.rsa.RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> pyformlang.rsa.RecursiveAutomaton:
    return pyformlang.rsa.RecursiveAutomaton.from_text(ebnf)
