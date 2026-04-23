def _copy_parent_node(state, orientation, parent_weight, child_weight, record_new_node=True):
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
    state.tree.add_edge(internal_node.node_id, parent_leaf.node_id, weight=float(parent_weight))
    state.tree.add_edge(internal_node.node_id, child_leaf.node_id, weight=float(child_weight))

    if record_new_node:
        state.new_nodes[internal_node] = (parent_leaf, child_leaf)

    return internal_node


def copy_parent_internal_node(state, orientation):
    return _copy_parent_node(
        state,
        orientation,
        parent_weight=0.0,
        child_weight=state.D[orientation.parent_idx, orientation.child_idx],
    )


def copy_parent_equal_weight_internal_node(state, orientation):
    weight = state.D[orientation.parent_idx, orientation.child_idx]
    return _copy_parent_node(
        state,
        orientation,
        parent_weight=weight,
        child_weight=weight,
    )


def copy_parent_without_new_node_record(state, orientation):
    return _copy_parent_node(
        state,
        orientation,
        parent_weight=0.0,
        child_weight=state.D[orientation.parent_idx, orientation.child_idx],
        record_new_node=False,
    )


def anticentral_weighted_copy_parent_node(state, orientation):
    return copy_parent_equal_weight_internal_node(state, orientation)


__all__ = [
    "anticentral_weighted_copy_parent_node",
    "copy_parent_equal_weight_internal_node",
    "copy_parent_internal_node",
    "copy_parent_without_new_node_record",
]
