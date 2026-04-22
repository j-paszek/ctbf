import random
from dataclasses import dataclass
from typing import Callable

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class PairChoice:
    i: int
    j: int
    score: object = None


@dataclass(frozen=True)
class Orientation:
    parent_idx: int
    child_idx: int


@dataclass
class ReconstructionState:
    D: np.ndarray
    D_full: np.ndarray
    node_list: list
    origin_index: dict
    tree: nx.DiGraph
    new_nodes: dict
    next_id: int
    rng: random.Random


PairSelector = Callable[[ReconstructionState], PairChoice]
AncestorSelector = Callable[[ReconstructionState, PairChoice], Orientation]
MergeStrategy = Callable[[ReconstructionState, Orientation], object]
DistanceUpdate = Callable[[ReconstructionState, Orientation, object], None]
RootStrategy = Callable[[ReconstructionState], tuple]


def initialize_reconstruction_state(dist_matrix, cells, max_id, seed=7, existing_tree=None):
    D_full = dist_matrix.copy().astype(float)
    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    node_list = [cells[i] for i in range(len(cells))]

    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    return ReconstructionState(
        D=D,
        D_full=D_full,
        node_list=node_list,
        origin_index={node: idx for idx, node in enumerate(node_list)},
        tree=tree,
        new_nodes={},
        next_id=max_id + 1,
        rng=random.Random(seed),
    )


def copy_parent_internal_node(state, orientation):
    parent_leaf = state.node_list[orientation.parent_idx]
    child_leaf = state.node_list[orientation.child_idx]

    internal_node = type(parent_leaf)(
        genome=parent_leaf.genome,
        node_id=state.next_id,
        cell_id=parent_leaf.cell_id,
    )
    state.next_id += 1
    state.origin_index[internal_node] = state.origin_index[parent_leaf]

    state.tree.add_node(
        internal_node.node_id,
        genome=internal_node.genome,
        cell_id=internal_node.cell_id,
    )
    state.tree.add_edge(internal_node.node_id, parent_leaf.node_id, weight=0.0)
    state.tree.add_edge(
        internal_node.node_id,
        child_leaf.node_id,
        weight=float(state.D[orientation.parent_idx, orientation.child_idx]),
    )
    state.new_nodes[internal_node] = (parent_leaf, child_leaf)
    return internal_node


def drop_child_keep_parent_update(state, orientation, internal_node):
    n = len(state.D)
    keep_indices = [k for k in range(n) if k != orientation.child_idx]
    state.D = state.D[np.ix_(keep_indices, keep_indices)]
    state.node_list[orientation.parent_idx] = internal_node
    state.node_list.pop(orientation.child_idx)


def remaining_lineage_root(state):
    root = state.node_list[0]
    return state.tree, state.new_nodes, root.cell_id


def run_agglomerative_reconstruction(
    dist_matrix,
    cells,
    max_id,
    *,
    seed=7,
    existing_tree=None,
    pair_selector: PairSelector,
    ancestor_selector: AncestorSelector,
    merge_strategy: MergeStrategy = copy_parent_internal_node,
    distance_update: DistanceUpdate = drop_child_keep_parent_update,
    root_strategy: RootStrategy = remaining_lineage_root,
):
    state = initialize_reconstruction_state(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
    )

    while len(state.node_list) > 1:
        pair = pair_selector(state)
        orientation = ancestor_selector(state, pair)
        internal_node = merge_strategy(state, orientation)
        distance_update(state, orientation, internal_node)

    return root_strategy(state)
