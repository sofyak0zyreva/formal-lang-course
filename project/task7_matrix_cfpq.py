from pyformlang.cfg import Variable, CFG
import networkx as nx
from typing import Dict, Set
from scipy.sparse import csr_matrix

from project.task6_cfpq import cfg_to_weak_normal_form, group_productions


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> set[tuple[int, int]]:
    cfg_wnf = cfg_to_weak_normal_form(cfg)

    index_to_node = list(graph.nodes())
    node_to_index = {node: idx for idx, node in enumerate(index_to_node)}
    n = graph.number_of_nodes()

    bool_matrices: Dict[Variable, csr_matrix] = {
        var: csr_matrix((n, n), dtype=bool) for var in cfg_wnf.variables
    }

    term_prods, epsilon_prods, binary_prods = group_productions(cfg_wnf)

    for u, v, label in graph.edges(data="label"):
        if label in term_prods:
            for A in term_prods[label]:
                bool_matrices[A][node_to_index[u], node_to_index[v]] = True

    for A in epsilon_prods:
        for i in range(n):
            bool_matrices[A][i, i] = True

    changed = True
    while changed:
        changed = False
        for A, B, C in binary_prods:
            product = bool_matrices[B] @ bool_matrices[C]
            diff = product - product.multiply(bool_matrices[A])
            if diff.count_nonzero() > 0:
                bool_matrices[A] = bool_matrices[A] + diff
                changed = True

    s = cfg_wnf.start_symbol
    all_pairs = zip(*bool_matrices[s].nonzero())

    result = {
        (index_to_node[i], index_to_node[j])
        for i, j in all_pairs
        if (not start_nodes or index_to_node[i] in start_nodes)
        and (not final_nodes or index_to_node[j] in final_nodes)
    }

    return result
