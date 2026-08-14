PARENT_REUSE_INTERNAL_NODE_IDS_CONTEXT_KEY = "parent_reuse_internal_node_ids"


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


def reuse_or_create_parent_internal_node(state, orientation):
    """Reuse one inferred occurrence when an active parent gains more children.

    Pairwise agglomeration normally wraps every selected pair in another copied
    parent occurrence and therefore forces a binary ladder.  This strategy
    creates that occurrence only the first time an active observed lineage is
    used as a parent.  If the resulting active inferred occurrence is selected
    as parent again, the next component is attached directly to it.  The
    inferred occurrence can consequently represent an unresolved multifurcation.

    The copied parent edge has weight zero because both occurrences carry the
    same CNP.  Output-family projection, when required, is deliberately handled
    by the caller after reconstruction rather than by this merge operation.
    """
    parent = state.node_list[orientation.parent_idx]
    child = state.node_list[orientation.child_idx]
    reusable_node_ids = state.context.setdefault(
        PARENT_REUSE_INTERNAL_NODE_IDS_CONTEXT_KEY,
        set(),
    )

    if parent.node_id in reusable_node_ids:
        state.tree.add_edge(
            parent.node_id,
            child.node_id,
            weight=float(state.D[orientation.parent_idx, orientation.child_idx]),
        )
        state.new_nodes[parent].append(child)
        return parent

    internal_node = _copy_parent_node(
        state,
        orientation,
        parent_weight=0.0,
        child_weight=state.D[orientation.parent_idx, orientation.child_idx],
    )
    state.new_nodes[internal_node] = list(state.new_nodes[internal_node])
    reusable_node_ids.add(internal_node.node_id)
    return internal_node


__all__ = [
    "PARENT_REUSE_INTERNAL_NODE_IDS_CONTEXT_KEY",
    "anticentral_weighted_copy_parent_node",
    "copy_parent_equal_weight_internal_node",
    "copy_parent_internal_node",
    "copy_parent_without_new_node_record",
    "reuse_or_create_parent_internal_node",
]
