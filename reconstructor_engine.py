import random
from dataclasses import dataclass, field
from typing import Callable

import networkx as nx
import numpy as np

from reconstructor_distance_update import drop_child_keep_parent_update
from reconstructor_merge import copy_parent_internal_node


@dataclass(frozen=True)
class PairChoice:
    i: int
    j: int
    score: object = None
    metadata: dict = field(default_factory=dict)


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
    context: dict = field(default_factory=dict)


PairSelector = Callable[[ReconstructionState], PairChoice]
AncestorSelector = Callable[[ReconstructionState, PairChoice], Orientation]
MergeStrategy = Callable[[ReconstructionState, Orientation], object]
DistanceUpdate = Callable[[ReconstructionState, Orientation, object], None]
RootStrategy = Callable[[ReconstructionState], tuple]
StateConfigurator = Callable[[ReconstructionState], None]


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


def remaining_lineage_root(state):
    root = state.node_list[0]
    # The public reconstruction API returns graph identity, not biological
    # state identity.  Copied-parent internal occurrences deliberately retain
    # their parent's ``cell_id`` while receiving a fresh ``node_id``; returning
    # the former can therefore name a non-root observed occurrence.
    return state.tree, state.new_nodes, root.node_id


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
    configure_state: StateConfigurator | None = None,
):
    state = initialize_reconstruction_state(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
    )
    if configure_state is not None:
        configure_state(state)

    while len(state.node_list) > 1:
        pair = pair_selector(state)
        orientation = ancestor_selector(state, pair)
        internal_node = merge_strategy(state, orientation)
        distance_update(state, orientation, internal_node)

    return root_strategy(state)
