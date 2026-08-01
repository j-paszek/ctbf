from dataclasses import dataclass
from typing import Callable

from reconstructor_algorithms import (
    make_nj_full_cps_variant,
    make_nj_full_variant,
    make_nj_hybrid_inv_cent_variant,
    make_nj_hybrid_variant,
    neighbor_joining_adaptive_centrality,
    neighbor_joining_adaptive_centrality_nonlinear,
    neighbor_joining_adaptive_centrality_reversed,
    new_alg,
    neighbor_joining_baseline,
    neighbor_joining_classical,
    neighbor_joining_hybrid_anticentral_adaptive_v3,
    neighbor_joining_hybrid_anticentral_adaptive_v3_plausible,
    neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony,
    neighbor_joining_hybrid_anticentral_adaptive_v3_skip_unplausible,
    neighbor_joining_hybrid_anticentral_opt,
    neighbor_joining_hybrid_opt,
    neighbor_joining_hybrid_opt_adaptive,
    neighbor_joining_hybrid_opt_refined,
    neighbor_joining_hybrid_opt_v2,
    rooted_labeled_nj,
    temporal_cnp_arborescence,
    temporal_cnp_arborescence_no_time,
)


@dataclass(frozen=True)
class ReconstructionAlgorithmSpec:
    """
    Stable registry entry for a reconstruction algorithm.

    `name` is the stable selection/display id. The position of an entry in
    `LEGACY_ALGORITHM_SPECS` is also a compatibility surface for historical
    benchmark filenames such as `0rec.csv`; do not reorder legacy entries.
    """
    name: str
    builder: Callable
    legacy: bool = True

    @property
    def stable_id(self):
        return self.name

    def build(self):
        algorithm = self.builder()
        if getattr(algorithm, "__name__", self.name) != self.name:
            algorithm.__name__ = self.name
        return algorithm


def _constant_algorithm(algorithm):
    return lambda: algorithm


LEGACY_ALGORITHM_SPECS = [
    ReconstructionAlgorithmSpec("neighbor_joining_baseline", _constant_algorithm(neighbor_joining_baseline)),
    ReconstructionAlgorithmSpec("neighbor_joining_full_full", lambda: make_nj_full_variant(True)),
    ReconstructionAlgorithmSpec("neighbor_joining_full_partial", lambda: make_nj_full_variant(False)),
    ReconstructionAlgorithmSpec("neighbor_joining_full_cps_full", lambda: make_nj_full_cps_variant(True)),
    ReconstructionAlgorithmSpec("neighbor_joining_full_cps_partial", lambda: make_nj_full_cps_variant(False)),
    ReconstructionAlgorithmSpec("neighbor_joining_hybrid_full", lambda: make_nj_hybrid_variant(True)),
    ReconstructionAlgorithmSpec("neighbor_joining_hybrid_partial", lambda: make_nj_hybrid_variant(False)),
    ReconstructionAlgorithmSpec(
        "neighbor_joining_hybrid_inverse_centrality_full",
        lambda: make_nj_hybrid_inv_cent_variant(True),
    ),
    ReconstructionAlgorithmSpec(
        "neighbor_joining_hybrid_inverse_centrality_partial",
        lambda: make_nj_hybrid_inv_cent_variant(False),
    ),
    ReconstructionAlgorithmSpec(
        "neighbor_joining_adaptive_centrality",
        _constant_algorithm(neighbor_joining_adaptive_centrality),
    ),
    ReconstructionAlgorithmSpec(
        "neighbor_joining_adaptive_centrality_nonlinear",
        _constant_algorithm(neighbor_joining_adaptive_centrality_nonlinear),
    ),
    ReconstructionAlgorithmSpec(
        "neighbor_joining_adaptive_centrality_reversed",
        _constant_algorithm(neighbor_joining_adaptive_centrality_reversed),
    ),
    ReconstructionAlgorithmSpec("neighbor_joining_hybrid_opt", _constant_algorithm(neighbor_joining_hybrid_opt)),
    ReconstructionAlgorithmSpec(
        "neighbor_joining_hybrid_opt_adaptive",
        _constant_algorithm(neighbor_joining_hybrid_opt_adaptive),
    ),
    ReconstructionAlgorithmSpec("neighbor_joining_hybrid_opt_v2", _constant_algorithm(neighbor_joining_hybrid_opt_v2)),
    ReconstructionAlgorithmSpec(
        "neighbor_joining_hybrid_opt_refined",
        _constant_algorithm(neighbor_joining_hybrid_opt_refined),
    ),
    ReconstructionAlgorithmSpec(
        "neighbor_joining_hybrid_anticentral_opt",
        _constant_algorithm(neighbor_joining_hybrid_anticentral_opt),
    ),
    ReconstructionAlgorithmSpec(
        "neighbor_joining_hybrid_anticentral_adaptive_v3",
        _constant_algorithm(neighbor_joining_hybrid_anticentral_adaptive_v3),
    ),
    ReconstructionAlgorithmSpec(
        "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible",
        _constant_algorithm(neighbor_joining_hybrid_anticentral_adaptive_v3_plausible),
    ),
    ReconstructionAlgorithmSpec(
        "neighbor_joining_hybrid_anticentral_adaptive_v3_skip_unplausible",
        _constant_algorithm(neighbor_joining_hybrid_anticentral_adaptive_v3_skip_unplausible),
    ),
    ReconstructionAlgorithmSpec(
        "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
        _constant_algorithm(neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony),
    ),
]


EXPERIMENTAL_ALGORITHM_SPECS = [
    ReconstructionAlgorithmSpec("new_alg", _constant_algorithm(new_alg), legacy=False),
]


PUBLICATION_ALGORITHM_SPECS = [
    ReconstructionAlgorithmSpec(
        "neighbor_joining_classical",
        _constant_algorithm(neighbor_joining_classical),
        legacy=False,
    ),
    ReconstructionAlgorithmSpec(
        "rooted_labeled_nj",
        _constant_algorithm(rooted_labeled_nj),
        legacy=False,
    ),
    ReconstructionAlgorithmSpec(
        "temporal_cnp_arborescence",
        _constant_algorithm(temporal_cnp_arborescence),
        legacy=False,
    ),
    ReconstructionAlgorithmSpec(
        "temporal_cnp_arborescence_no_time",
        _constant_algorithm(temporal_cnp_arborescence_no_time),
        legacy=False,
    ),
]


def build_algorithms(specs):
    return [spec.build() for spec in specs]


__all__ = [
    "EXPERIMENTAL_ALGORITHM_SPECS",
    "LEGACY_ALGORITHM_SPECS",
    "PUBLICATION_ALGORITHM_SPECS",
    "ReconstructionAlgorithmSpec",
    "build_algorithms",
]
