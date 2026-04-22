import numpy as np


# ============================================================
#  BIOLOGICAL PLAUSIBILITY
# ============================================================
def _is_biologically_plausible_ancestor(ancestor, descendant):
    """
    Returns True if 'ancestor' could biologically be the parent of 'descendant'.

    Constraint:
    - ancestor cannot generate descendant if ancestor has CN=0 at a locus
      where descendant has CN>0 (i.e. a gain from 0 -> positive is disallowed).
    """
    return not np.any((ancestor.genome == 0) & (descendant.genome > 0))


def _is_biologically_plausible_pair(x, y):
    """
    A pair (x, y) is biologically plausible if at least one direction
    (x->y or y->x) is biologically possible.

    This means:
    - keep this pair if x can be parent of y OR y can be parent of x.
    """
    return (_is_biologically_plausible_ancestor(x, y) or
            _is_biologically_plausible_ancestor(y, x))


# ============================================================
#  PARENT SELECTION - THE ROOT
# ============================================================
def _final_parent_choice_full_matrix(x, y, D_full, origin_index, rng, select_ancestor_func):
    """
    Decide the final parent-child direction for the last two lineages
    using the ORIGINAL (full) distance matrix D_full.

    x, y : the two remaining nodes (Genotype-like objects)
    origin_index : dict mapping node -> original index in D_full
    """
    ix = origin_index[x]
    iy = origin_index[y]

    parent_idx, child_idx = select_ancestor_func(
        D_full, ix, iy, rng, larger_is_more_central=False
    )

    # Map index choice back to node objects
    if parent_idx == ix:
        return x, y
    else:
        return y, x


# ============================================================
#  PARENT SELECTION
# ============================================================
def _choose_parent_full_nj(D, i, j, rng, larger_is_more_central):
    """
    NJ-parent rule: more central -> parent
    """
    centrality = D.sum(axis=1)
    c_i, c_j = centrality[i], centrality[j]

    if c_i < c_j:
        return i, j
    if c_j < c_i:
        return j, i

    # Centrality tie:
    return (i, j) if rng.random() < 0.5 else (j, i)


def _choose_parent_hybrid_inv_centrality(D, i, j, rng, larger_is_more_central=False, epsilon=1e-6):
    """
    Parent = node with larger inverse-distance centrality.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        invD = 1.0 / (D + epsilon)
        np.fill_diagonal(invD, 0.0)
        centrality = invD.sum(axis=1)

    c_i, c_j = centrality[i], centrality[j]

    if c_i > c_j:
        return i, j
    if c_j > c_i:
        return j, i

    # tie
    return (i, j) if rng.random() < 0.5 else (j, i)


def _choose_parent_with_full_matrix(
    D_full, origin_index, node_list, i, j, rng, select_ancestor_func
):
    """
    Use the FULL distance matrix to decide which of node_list[i], node_list[j]
    should be the parent.

    Returns (parent_idx, child_idx) in *current* node_list indexing.
    """
    x = node_list[i]
    y = node_list[j]

    ix = origin_index[x]
    iy = origin_index[y]

    parent_full_idx, child_full_idx = select_ancestor_func(
        D_full, ix, iy, rng, larger_is_more_central=False
    )

    if parent_full_idx == ix:
        return i, j
    else:
        return j, i


def _choose_parent_with_plausibility_fallback(
    D,
    D_full,
    origin_index,
    node_list,
    i,
    j,
    rng,
    select_ancestor_func,
    full_information=False,
):
    """
    Orient a selected pair using biological plausibility when it decides.

    If exactly one direction is biologically plausible, force that direction.
    If both or neither direction is plausible, fall back to the algorithm's
    configured ancestor-selection rule. This preserves the soft-fallback
    behavior used by the plausible NJ variants.
    """
    x = node_list[i]
    y = node_list[j]

    can_x_parent = _is_biologically_plausible_ancestor(x, y)
    can_y_parent = _is_biologically_plausible_ancestor(y, x)

    if can_x_parent and not can_y_parent:
        return i, j

    if can_y_parent and not can_x_parent:
        return j, i

    if full_information:
        return _choose_parent_with_full_matrix(
            D_full, origin_index, node_list, i, j, rng, select_ancestor_func
        )

    return select_ancestor_func(D, i, j, rng, larger_is_more_central=False)
