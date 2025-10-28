from typing import Dict, Tuple
from pyformlang.cfg import Production, Variable, Terminal, CFG, Epsilon
import pyformlang.cfg as pcfg
import networkx as nx


def cfg_to_weak_normal_form(cfg: pcfg.CFG) -> pcfg.CFG:
    nullable_vars = cfg.get_nullable_symbols()
    cfg_nf = cfg.to_normal_form()
    epsilon_prods = {Production(var, [Epsilon()]) for var in nullable_vars}
    return CFG(
        cfg_nf.variables,
        cfg_nf.terminals | {Epsilon()},
        cfg_nf.start_symbol,
        cfg_nf.productions | epsilon_prods,
    )


def group_productions(cfg_wnf: pcfg.CFG):
    term_prods: Dict[str, set[Variable]] = {}
    epsilon_prods: set[Variable] = set()
    binary_prods: set[Tuple[Variable, Variable, Variable]] = set()
    for p in cfg_wnf.productions:
        head = p.head
        body = list(p.body)
        if len(body) == 0:
            epsilon_prods.add(head)
        elif len(body) == 1 and isinstance(body[0], Terminal):
            term_prods.setdefault(body[0].value, set()).add(head)
        elif (
            len(body) == 2
            and isinstance(body[0], Variable)
            and isinstance(body[1], Variable)
        ):
            binary_prods.add((head, body[0], body[1]))
    return term_prods, epsilon_prods, binary_prods


def hellings_initialize(
    graph: nx.DiGraph,
    term_prods: Dict[str, set[Variable]],
    epsilon_prods: set[Variable],
) -> Dict[Variable, set[Tuple[int, int]]]:
    reached = {}

    for u, v, label in graph.edges(data="label"):
        label = label if label is not None else "$"
        if label in term_prods:
            for A in term_prods[label]:
                reached.setdefault(A, set()).add((int(u), int(v)))

    for A in epsilon_prods:
        for v in graph.nodes():
            reached.setdefault(A, set()).add((int(v), int(v)))

    return reached


def hellings_based_cfpq(
    cfg: pcfg.CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    cfg_wnf = cfg_to_weak_normal_form(cfg)

    term_prods, epsilon_prods, binary_prods = group_productions(cfg_wnf)
    reached = hellings_initialize(graph, term_prods, epsilon_prods)

    changed = True
    while changed:
        changed = False
        for A, B, C in binary_prods:
            if B in reached and C in reached:
                b_pairs = list(reached[B])
                c_pairs = list(reached[C])
                for u, v1 in b_pairs:
                    for v2, w in c_pairs:
                        if v1 == v2 and (u, w) not in reached.setdefault(A, set()):
                            reached[A].add((u, w))
                            changed = True

    result = set()
    start_symbol = cfg_wnf.start_symbol
    if start_symbol in reached:
        for u, v in reached[start_symbol]:
            if (not start_nodes or u in start_nodes) and (
                not final_nodes or v in final_nodes
            ):
                result.add((u, v))

    return result
