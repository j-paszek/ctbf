from reconstructor_biopsy_blocks import (
    assign_compatible_node_ids,
    assign_new_node_levels,
    build_final_distance_matrix,
    copy_reconstruction_cell_lists,
    initialize_biopsy_tree,
    make_id_to_index,
    make_unique_node_counter,
    normalize_biopsy_guided_config,
    reconstruct_biopsy_layers,
)
from reconstructor_utils import parse_distance_matrix
from reconstructor_temporal import (
    uses_directed_distance_input,
    uses_ordered_occurrence_input,
)
from distance_semantics import (
    validate_distance_label_coverage,
    validate_distance_matrix,
)


def build_evolution_tree_impl(
    cell_lists,
    seed=7,
    dist_matrix_path=None,
    r=2,
    only_nj=False,
    inids=None,
    indm=None,
    directed_distance_bundle=None,
    neighbor_joining=None,
    biopsy_guided_config=None,
):
    cell_lists = copy_reconstruction_cell_lists(cell_lists)

    if dist_matrix_path:
        ids, full_dist_matrix = parse_distance_matrix(dist_matrix_path)
    elif inids is not None and indm is not None:
        ids, full_dist_matrix = inids, indm
    else:
        print("Please provide either dist_matrix_path or inids and indm")
        return None

    ids, full_dist_matrix = validate_distance_matrix(ids, full_dist_matrix)
    observed_ids = []
    seen_observed_ids = set()
    for cell_list in cell_lists:
        for cell in cell_list:
            cell_id = cell.cell_id
            try:
                if cell_id not in seen_observed_ids:
                    seen_observed_ids.add(cell_id)
                    observed_ids.append(cell_id)
            except TypeError as exc:
                raise ValueError("Observed CNP labels must be hashable.") from exc
    validate_distance_label_coverage(ids, observed_ids, allow_extra=True)

    if uses_ordered_occurrence_input(neighbor_joining):
        if only_nj:
            raise ValueError(
                "Occurrence-level reconstruction does not use only_nj pooling; "
                "select temporal_cnp_arborescence_no_time for the exact order ablation."
            )
        if biopsy_guided_config is not None:
            raise ValueError(
                "Occurrence-level reconstruction cannot be combined with a "
                "biopsy-guided interpolation preset."
            )
        requires_directed = uses_directed_distance_input(neighbor_joining)
        if requires_directed and directed_distance_bundle is None:
            raise ValueError(
                f"{neighbor_joining.__name__} requires a DirectedDistanceBundle."
            )
        if directed_distance_bundle is not None and not requires_directed:
            raise ValueError(
                "Directed distance evidence may be passed only to an algorithm "
                "that explicitly declares directed-distance support."
            )
        algorithm_kwargs = {"seed": seed}
        if requires_directed:
            algorithm_kwargs["directed_distance_bundle"] = directed_distance_bundle
        tree, _new_nodes, final_root = neighbor_joining(
            full_dist_matrix,
            cell_lists,
            ids,
            **algorithm_kwargs,
        )
        node_levels = {
            node_id: data["biopsy_level"]
            for node_id, data in tree.nodes(data=True)
        }
        return tree, node_levels, final_root

    if directed_distance_bundle is not None:
        raise ValueError(
            "Directed distance evidence may be passed only to an ordered "
            "algorithm that explicitly declares directed-distance support."
        )

    biopsy_guided_config = normalize_biopsy_guided_config(biopsy_guided_config)

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
