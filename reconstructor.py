from reconstructor_algorithms import *  # noqa: F403
from reconstructor_algorithms import __all__ as _ALGORITHM_EXPORTS
from reconstructor_biopsy_blocks import (
    BiopsyGuidedConfig,
    BiopsySubtreeConfig,
    default_biopsy_guided_config,
    make_binarized_group_attachment_strategy,
    select_anticentral_candidate,
)
from reconstructor_biopsy_presets import (
    BIOPSY_GUIDED_PRESETS,
    make_anticentral_binarized_biopsy_guided_config,
    make_anticentral_tie_biopsy_guided_config,
    make_binarized_biopsy_guided_config,
    make_default_biopsy_guided_config,
    resolve_biopsy_guided_config,
)
from reconstructor_biopsy_guided import build_evolution_tree_impl
from reconstructor_utils import visualize_tree_plotly
from simulator import Genotype


def build_evolution_tree(cell_lists, seed=7, dist_matrix_path=None, r=2, only_nj=False, inids=None, indm=None,
                         distance_matrix=None, directed_distance_bundle=None,
                         neighbor_joining=neighbor_joining_standard, biopsy_guided_config=None):  # noqa: F405
    if distance_matrix is not None:
        if any(value is not None for value in (
            dist_matrix_path,
            inids,
            indm,
            directed_distance_bundle,
        )):
            raise ValueError("Pass either distance_matrix or legacy distance matrix arguments, not both.")
        distance_kwargs = distance_matrix.build_tree_kwargs()
        dist_matrix_path = distance_kwargs.get("dist_matrix_path")
        inids = distance_kwargs.get("inids")
        indm = distance_kwargs.get("indm")
        directed_distance_bundle = distance_kwargs.get("directed_distance_bundle")

    return build_evolution_tree_impl(
        cell_lists,
        seed=seed,
        dist_matrix_path=dist_matrix_path,
        r=r,
        only_nj=only_nj,
        inids=inids,
        indm=indm,
        directed_distance_bundle=directed_distance_bundle,
        neighbor_joining=neighbor_joining,
        biopsy_guided_config=biopsy_guided_config,
    )


__all__ = list(_ALGORITHM_EXPORTS) + [
    "BiopsyGuidedConfig",
    "BiopsySubtreeConfig",
    "BIOPSY_GUIDED_PRESETS",
    "build_evolution_tree",
    "default_biopsy_guided_config",
    "make_anticentral_binarized_biopsy_guided_config",
    "make_anticentral_tie_biopsy_guided_config",
    "make_binarized_group_attachment_strategy",
    "make_binarized_biopsy_guided_config",
    "make_default_biopsy_guided_config",
    "resolve_biopsy_guided_config",
    "select_anticentral_candidate",
    "visualize_tree_plotly",
]


if __name__ == '__main__':
    # CNPs here do not influence distances only for checking if descendant has x>0 where ancestor has x=0
    cell_lists = [
        [Genotype([2, 0, 1], 1), Genotype([1, 1, 1], 2)],
        [Genotype([2, 1, 1], 3), Genotype([1, 2, 0], 4)]
    ]
    cell_lists1 = [
        [Genotype([2, 2, 1], 1), Genotype([1, 1, 1], 2)],
        [Genotype([2, 1, 1], 3), Genotype([1, 2, 0], 4)]
    ]

    # # 3->2
    # tree, a, b = build_evolution_tree(cell_lists, "test/data/dm/distance_matrix.txt", r=2)
    # visualize_tree_plotly(tree, a)
    # # 3->1
    # tree, a, b = build_evolution_tree(cell_lists1, "test/data/dm/distance_matrix.txt", r=2)
    # visualize_tree_plotly(tree, a)
    # 3->2, 4->2
    tree, a, _ = build_evolution_tree(cell_lists, "test/data/dm/distance_matrix.txt", r=4)
    visualize_tree_plotly(tree, a)
