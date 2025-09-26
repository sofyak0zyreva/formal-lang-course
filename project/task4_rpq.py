from networkx import MultiDiGraph
from scipy.sparse import vstack, csr_matrix

from project.task2_automata_conversions import graph_to_nfa, regex_to_dfa
from project.task3_adjacency_matrix import AdjacencyMatrixFA


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    graph_am = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_am = AdjacencyMatrixFA(regex_to_dfa(regex))
    alphabet = regex_am.alphabet & graph_am.alphabet
    m = graph_am.size
    n = regex_am.size
    ind_g = graph_am.state_index
    ind_r = regex_am.state_index

    blocks = []
    for s in graph_am.start_states:
        block = csr_matrix((m, n), dtype=bool)
        for q in regex_am.start_states:
            block[ind_g[s], ind_r[q]] = True
        blocks.append(block)
    front = vstack(blocks)
    visited = front

    graph_matrices_transposed = {
        symbol: graph_am.decomposed_adj_matrix[symbol].T.tocsr() for symbol in alphabet
    }
    regex_matrices = {
        symbol: regex_am.decomposed_adj_matrix[symbol].tocsr() for symbol in alphabet
    }
    num = len(graph_am.start_states)

    while front.nnz > 0:
        new_front = {}
        for symbol in alphabet:
            new_front[symbol] = vstack(
                [
                    (graph_matrices_transposed[symbol] @ front[i * m : (i + 1) * m])
                    for i in range(num)
                ]
            )
            new_front[symbol] = new_front[symbol] @ regex_matrices[symbol]
        front = sum(new_front.values()) > visited
        visited = visited + front

    state_pairs = set()
    for i, s_from in enumerate(graph_am.start_states):
        block = visited[i * m : (i + 1) * m]
        for s_to in graph_am.final_states:
            row = block.getrow(ind_g[s_to])
            for q in regex_am.final_states:
                if row[0, ind_r[q]]:
                    state_pairs.add((s_from.value, s_to.value))
    return state_pairs
