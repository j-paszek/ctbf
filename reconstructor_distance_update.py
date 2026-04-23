import numpy as np


ANTICENTRAL_V3_CONTEXT_KEY = "anticentral_v3_centrality"


def drop_child_keep_parent_update(state, orientation, internal_node):
    n = len(state.D)
    keep_indices = [k for k in range(n) if k != orientation.child_idx]
    state.D = state.D[np.ix_(keep_indices, keep_indices)]
    state.node_list[orientation.parent_idx] = internal_node
    state.node_list.pop(orientation.child_idx)


def anticentral_v3_distance_update(state, orientation, internal_node):
    n = len(state.D)
    c = state.context[ANTICENTRAL_V3_CONTEXT_KEY]
    keep_indices = [k for k in range(n) if k != orientation.child_idx]

    state.D = state.D[np.ix_(keep_indices, keep_indices)]
    state.node_list[orientation.parent_idx] = internal_node
    state.node_list.pop(orientation.child_idx)

    c[orientation.parent_idx] = np.mean([c[orientation.parent_idx], c[orientation.child_idx]])
    state.context[ANTICENTRAL_V3_CONTEXT_KEY] = np.delete(c, orientation.child_idx)


__all__ = [
    "anticentral_v3_distance_update",
    "drop_child_keep_parent_update",
]
