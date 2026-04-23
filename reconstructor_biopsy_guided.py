from reconstructor_biopsy_blocks import (
    assign_compatible_node_ids,
    assign_new_node_levels,
    build_final_distance_matrix,
    initialize_biopsy_tree,
    make_id_to_index,
    make_unique_node_counter,
    normalize_biopsy_guided_config,
    reconstruct_biopsy_layers,
)
from reconstructor_utils import parse_distance_matrix


_extend_biopsies = normalize_biopsy_guided_config().level_extender


def build_evolution_tree_impl(
    cell_lists,
    seed=7,
    dist_matrix_path=None,
    r=2,
    only_nj=False,
    inids=None,
    indm=None,
    neighbor_joining=None,
    biopsy_guided_config=None,
):
    biopsy_guided_config = normalize_biopsy_guided_config(biopsy_guided_config)

    if dist_matrix_path:
        ids, full_dist_matrix = parse_distance_matrix(dist_matrix_path)
    elif inids is not None and indm is not None:
        ids, full_dist_matrix = inids, indm
    else:
        print("Please provide either dist_matrix_path or inids and indm")
        return None

    cell_lists = biopsy_guided_config.level_extender(cell_lists)

    id_to_index = make_id_to_index(ids)
    unique_node_counter = make_unique_node_counter(ids)
    assign_compatible_node_ids(cell_lists)
    tree, node_levels = initialize_biopsy_tree(cell_lists, unique_node_counter, only_nj=only_nj)

    if not only_nj:
        reconstruct_biopsy_layers(
            cell_lists,
            tree,
            node_levels,
            full_dist_matrix,
            id_to_index,
            r,
            unique_node_counter,
            config=biopsy_guided_config,
        )

    final_cells = cell_lists[0]  # may contain same genotype 0 distance cells not merged due to appearance constraint
    if only_nj: # in NJ triggered by argument we assume no duplicate cell genotypes
        final_cells = biopsy_guided_config.only_nj_final_cell_selector(final_cells)

    dist_matrix = build_final_distance_matrix(final_cells, full_dist_matrix, id_to_index)

    max_id = next(unique_node_counter)
    tree, new_nodes, final_root = neighbor_joining(dist_matrix, final_cells, max_id, existing_tree=tree, seed=seed)
    assign_new_node_levels(new_nodes, node_levels)

    return tree, node_levels, final_root
