from typing import Dict, Set, Tuple
import pyformlang
from dataclasses import dataclass
import networkx as nx
import pyformlang.rsa
from pyformlang.finite_automaton import (
    State,
    Symbol,
)


@dataclass(frozen=True)
class RSMState:
    symbol: Symbol
    state: State


@dataclass(frozen=True)
class SPPFNode:
    state: RSMState
    node: int
    gss_node: "GSSNode"


class GSSNode:
    def __init__(self, state: RSMState, node: int):
        self.state = state
        self.node = node
        self.edges = {}
        self.stack = set()

    def pop(self, node: int) -> Set[SPPFNode]:
        if node in self.stack:
            return set()
        self.stack.add(node)
        return {
            SPPFNode(state, node, gss_node)
            for state, gss_nodes in self.edges.items()
            for gss_node in gss_nodes
        }

    # [self] ---state--> [gss_node]
    def add_edge(self, state: RSMState, gss_node: "GSSNode") -> Set[SPPFNode]:
        if gss_node in self.edges.setdefault(state, set()):
            return set()
        self.edges[state].add(gss_node)
        return {SPPFNode(state, node, gss_node) for node in self.stack}


@dataclass
class RSMStateData:
    # S is the symbol of our current rsm box
    # [state1] ---term--> [state2]
    # term, (S, state2)
    term_edges: Dict[str, RSMState]
    # [state1] ---S--> [state2],
    # nonterm, (called nonterm, its start state), (state2, S) -- call and return
    nonterm_edges: Dict[str, Tuple[RSMState, RSMState]]
    is_final: bool


class GLL:
    def __init__(self, rsm: pyformlang.rsa.RecursiveAutomaton, graph: nx.DiGraph):
        self.accept_gss_node = GSSNode(RSMState("$", "fin"), None)
        self.gss_nodes = {}
        self.pending_sppf_nodes: Set[SPPFNode] = set()
        self.added_sppf_nodes: Set[SPPFNode] = set()
        self.graph = graph
        self.rsm_start_state = RSMState(
            rsm.initial_label, rsm.boxes[rsm.initial_label].dfa.start_state.value
        )
        self.rsm_state_data: Dict[Symbol, Dict[State, RSMStateData]] = {
            symbol: {
                state: RSMStateData({}, {}, state in box.dfa.final_states)
                for state in box.dfa.to_networkx().nodes
            }
            for symbol, box in rsm.boxes.items()
        }
        # a ---label--> b ; a ---label--> c
        # outer_edges[a][label] = {b; c}
        self.outer_edges = {
            node: {label: set() for _, _, label in graph.edges(data="label") if label}
            for node in graph.nodes()
        }
        for node_from, node_to, label in graph.edges(data="label"):
            if label:
                self.outer_edges[node_from].setdefault(label, set()).add(node_to)
        # fill in term and nonterm edges of rsm
        for nonterm, box in rsm.boxes.items():
            for s_from, s_to, symbol in box.dfa.to_networkx().edges(data="label"):
                if symbol:
                    if Symbol(symbol) not in rsm.boxes:
                        self.rsm_state_data[nonterm][s_from].term_edges[symbol] = (
                            RSMState(nonterm, s_to)
                        )
                    else:
                        start_s_called = rsm.boxes[Symbol(symbol)].dfa.start_state
                        self.rsm_state_data[nonterm][s_from].nonterm_edges[symbol] = (
                            RSMState(symbol, start_s_called.value),
                            RSMState(nonterm, s_to),
                        )

    def add_sppf_nodes(self, sppf_nodes: Set[SPPFNode]):
        new_sppf_nodes = sppf_nodes - self.added_sppf_nodes
        self.added_sppf_nodes.update(new_sppf_nodes)
        self.pending_sppf_nodes.update(new_sppf_nodes)

    def process_popped_sppf_nodes(
        self, sppf_nodes: Set[SPPFNode], prev_sppf_node: SPPFNode
    ):
        processed_sppf_nodes = set()
        graph_nodes = set()
        for sppf_node in sppf_nodes:
            if sppf_node.gss_node == self.accept_gss_node:
                graph_node_from = prev_sppf_node.gss_node.node
                graph_node_to = sppf_node.node
                graph_nodes.add((graph_node_from, graph_node_to))
            else:
                processed_sppf_nodes.add(sppf_node)
        return processed_sppf_nodes, graph_nodes

    def gss_node(self, state: RSMState, node: int) -> GSSNode:
        if (state, node) not in self.gss_nodes:
            self.gss_nodes[(state, node)] = GSSNode(state, node)
        return self.gss_nodes[(state, node)]

    def cfpq(self, nodes_from: set[int], nodes_to: set[int]):
        if not nodes_from:
            nodes_from = set(self.graph.nodes())
        if not nodes_to:
            nodes_to = set(self.graph.nodes())
        result = set()
        for node in nodes_from:
            gss_node = self.gss_node(self.rsm_start_state, node)
            gss_node.add_edge(RSMState("$", "fin"), self.accept_gss_node)
            self.add_sppf_nodes({SPPFNode(self.rsm_start_state, node, gss_node)})
        while self.pending_sppf_nodes:
            sppf_node = self.pending_sppf_nodes.pop()
            rsm_state_data = self.rsm_state_data[sppf_node.state.symbol][
                sppf_node.state.state
            ]
            node = sppf_node.node
            for term, state in rsm_state_data.term_edges.items():
                if term in self.outer_edges[node]:
                    self.add_sppf_nodes(
                        {
                            SPPFNode(state, node_to, sppf_node.gss_node)
                            for node_to in self.outer_edges[node][term]
                        }
                    )
            for _, (called_state, ret_state) in rsm_state_data.nonterm_edges.items():
                gss_node_from = self.gss_node(called_state, sppf_node.node)
                sppf_nodes = gss_node_from.add_edge(ret_state, sppf_node.gss_node)
                processed_sppf_nodes, reachable = self.process_popped_sppf_nodes(
                    sppf_nodes, sppf_node
                )
                self.add_sppf_nodes(
                    {SPPFNode(called_state, sppf_node.node, gss_node_from)}
                )
                self.add_sppf_nodes(processed_sppf_nodes)
                result.update(reachable)
            if rsm_state_data.is_final:
                sppf_nodes = sppf_node.gss_node.pop(sppf_node.node)
                processed_sppf_nodes, reachable = self.process_popped_sppf_nodes(
                    sppf_nodes, sppf_node
                )
                self.add_sppf_nodes(processed_sppf_nodes)
                result.update(reachable)
        return {
            (graph_node_from, graph_node_to)
            for graph_node_from, graph_node_to in result
            if graph_node_to in nodes_to
        }


def gll_based_cfpq(
    rsm: pyformlang.rsa.RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    return GLL(rsm, graph).cfpq(start_nodes, final_nodes)
