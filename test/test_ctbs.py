import copy

import numpy as np
import pytest
from networkx.utils.misc import flatten
import os

from ctbs import (
    SuppliedDistanceProvider,
    run_single_test,
    use_cnp2cnp_to_compute_pairwise_distance,
    distance_matrix_from_biopsy,
)
from ctbs_utils import to_newick

import json
import networkx as nx
from networkx.readwrite import json_graph

import ctbs
from reconstructor import build_evolution_tree, visualize_tree_plotly
from reconstructor_algorithms import (
    neighbor_joining_baseline,
    neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony,
    neighbor_joining_hybrid_opt,
    temporal_cnp_arborescence,
)
from simulator import CancerCellEvolutionSimulator, Genotype

SHOW_FIGURES = False
TEST_DIR = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


def data_path(*parts):
    return os.path.join(TEST_DATA_DIR, *parts)


def get_sim_from_tree(tr):
    tree = tree_from_json(tr)
    sim = CancerCellEvolutionSimulator.from_tree(tree)
    return sim


def tree_from_json(tr_id):
    with open(data_path("tree_samples", str(tr_id) + ".json")) as f:
        data_loaded = json.load(f)
    return json_graph.node_link_graph(data_loaded, directed=True, edges="links")


def generate_biopsy_sets():
    c1 = Genotype(node_id=7, generation=6, cell_id=7, genome=[2, 0, 2, 2, 2, 2, 2, 4, 2, 2])
    c2 = Genotype(node_id=14, generation=8, cell_id=7, genome=[2, 0, 2, 2, 2, 2, 2, 4, 2, 2])
    c3 = Genotype(node_id=13, generation=8, cell_id=13, genome=[2, 2, 2, 2, 2, 2, 2, 4, 2, 2])
    biopsy_set1 = [[c1], [c2, c3]]  # cannot match 13 to 7
    njbs1 = [[c1, c2, c3]] # different input for NJ reconstruction
    c4 = Genotype(node_id=5, generation=6, cell_id=13, genome=[2, 2, 2, 2, 2, 2, 2, 4, 2, 2])
    biopsy_set2 = [[c4], [c2, c3]]  # should match 14 to 5
    njbs2 = [[c4, c2, c3]]
    return biopsy_set1, biopsy_set2, njbs1, njbs2


def generate_biopsy_sets_small():
    c1 = Genotype([2, 0, 1], 1)
    c2 = Genotype([1, 1, 1], 2)
    c3 = Genotype([2, 1, 1], 3)
    c4 = Genotype([1, 2, 0], 4)
    b = [[c1, c2], [c3, c4]]
    nj = [[c1, c2, c3, c4]]
    c5 = Genotype([2, 2, 1], 1)
    b1 = [[c5, c2], [c3, c4]]
    nj1 = [[c5, c2, c3, c4]]
    return b, b1, nj, nj1


def test_scalable_biopsy_samples_one_cell_from_nonempty_generation():
    sim = get_sim_from_tree(689)
    biopsy = sim.perform_biopsy(biopsy_size_scalable=0.5, generation=4, seed=689)
    assert [(cell.node_id, cell.cell_id) for cell in biopsy] == [(4, 1)]


def test_scalable_biopsy_keeps_empty_generation_empty():
    sim = get_sim_from_tree(689)
    biopsy = sim.perform_biopsy(biopsy_size_scalable=0.5, generation=99, seed=689)
    assert biopsy == []


def test_compare_dm_uses_occurrence_nodes_and_canonical_labels(tmp_path):
    tree = nx.DiGraph()
    tree.add_node(10, genome=[2], generation=0, cell_id=10)
    tree.add_node(20, genome=[3], generation=1, cell_id=20)
    tree.add_node(30, genome=[4], generation=2, cell_id=30)
    tree.add_edge(10, 20, events="duplication(pos=0, copies=1)")
    tree.add_edge(20, 30, events="duplication(pos=0, copies=1)")
    sim = CancerCellEvolutionSimulator.from_tree(tree)
    observations = [
        Genotype([3], node_id=20, generation=1, cell_id=1),
        Genotype([4], node_id=30, generation=2, cell_id=2),
    ]
    output_path = tmp_path / "truth_observations.phy"

    ids, matrix = ctbs._write_observed_truth_distance_matrix(
        sim,
        observations,
        output_path,
    )

    assert ids == [1, 2]
    assert np.array_equal(matrix, [[0.0, 1.0], [1.0, 0.0]])
    assert output_path.read_text().splitlines()[1].split()[0] == "1"


def test_compare_dm_rejects_ambiguous_repeated_cnp_occurrences(tmp_path):
    class UnusedSimulator:
        def to_distance_matrix(self, *args, **kwargs):
            pytest.fail("ambiguous truth diagnostic must fail before serialization")

    observations = [
        Genotype([2], node_id=20, generation=1, cell_id=7),
        Genotype([2], node_id=30, generation=2, cell_id=7),
    ]

    with pytest.raises(ValueError, match="same canonical CNP label"):
        ctbs._write_observed_truth_distance_matrix(
            UnusedSimulator(),
            observations,
            tmp_path / "ambiguous.phy",
        )


def test_simulator_distance_matrix_rejects_unknown_occurrence_node():
    tree = nx.DiGraph()
    tree.add_node(10, genome=[2], generation=0, cell_id=10)
    sim = CancerCellEvolutionSimulator.from_tree(tree)

    with pytest.raises(ValueError, match="unknown node ids"):
        sim.to_distance_matrix(node_list=[999], labels=[1])


@pytest.mark.parametrize(
    "algorithm",
    [
        neighbor_joining_baseline,
        neighbor_joining_hybrid_opt,
        neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony,
        temporal_cnp_arborescence,
    ],
    ids=lambda algorithm: algorithm.__name__,
)
def test_run_single_test_accepts_two_biopsy_cells_and_uses_actual_grf_roots(monkeypatch, algorithm):
    class TwoCellSimulator:
        def __init__(self):
            self.tree = nx.DiGraph()
            self.tree.add_node(0, genome=[2, 2], generation=0, cell_id=0)
            self.tree.add_node(5, genome=[2, 2], generation=1, cell_id=5)
            self.tree.add_node(7, genome=[3, 2], generation=2, cell_id=7)
            self.tree.add_edge(0, 5, events="")
            self.tree.add_edge(5, 7, events="duplication(pos=0, copies=1)")

        def perform_biopsy(
            self,
            generation,
            biopsy_size=0,
            biopsy_size_scalable=None,
            seed=None,
        ):
            if generation == 1:
                return [Genotype([2, 2], node_id=5, generation=1, cell_id=5)]
            if generation == 2:
                return [Genotype([3, 2], node_id=7, generation=2, cell_id=7)]
            return []

    seen_roots = []

    def actual_root(tree):
        roots = [node for node, indegree in tree.in_degree() if indegree == 0]
        assert len(roots) == 1
        return roots[0]

    def capture_grf(true_tree, true_root, reconstructed_tree, reconstructed_root):
        seen_roots.append(reconstructed_root)
        assert reconstructed_root == actual_root(reconstructed_tree)
        return 1.0

    monkeypatch.setattr(ctbs, "grf_tree", capture_grf)
    distance_provider = SuppliedDistanceProvider(
        ids=[5, 7],
        matrix=np.array([
            [0.0, 1.0],
            [1.0, 0.0],
        ]),
    )

    result = run_single_test(
        seed=7,
        simulator_with_loaded_tree=TwoCellSimulator(),
        biopsy_generations=[1, 2],
        biopsy_size_scalable=0.5,
        r_dist=4,
        distance_provider=distance_provider,
        reconstruction_algorithm=algorithm,
    )

    assert result is not None
    assert len(seen_roots) == 2


def test_reconstructor_no_connecting_within_distance():
    sim = get_sim_from_tree(689)
    bg = [4, 6, 8]
    common_params = {
        "seed": 689,
        "biopsy_size_scalable": 0.5
    }
    def observation(node_id):
        data = sim.tree.nodes[node_id]
        return Genotype(
            genome=data["genome"],
            node_id=node_id,
            generation=data["generation"],
            cell_id=data["cell_id"],
        )

    # Encode the reconstruction edge case directly; its result must not depend
    # on the exact samples emitted by a historical RNG implementation.
    biopsies_by_generation = {
        4: [observation(4)],
        6: [observation(7)],
        8: [observation(14), observation(13)],
    }
    cell_lists = [biopsies_by_generation[generation] for generation in bg]

    def explicit_biopsy(generation, **_kwargs):
        return copy.deepcopy(biopsies_by_generation[generation])

    sim.perform_biopsy = explicit_biopsy
    assert [[1], [7], [7, 13]] == [[c.cell_id for c in cl] for cl in cell_lists]
    # generation 4 now contributes one cell due the scalable-biopsy minimum.
    assert [[4], [7], [14, 13]] == [[c.node_id for c in cl] for cl in cell_lists]
    # cells 7 and 14 share the same genotype
    assert 0 == cell_lists[1][0].genome[1]  # cell 7 (oldest) has 0 in CNP (for i=1, cnp(i)=0)
                                            # [Genotype(ID=7, Gen=6, cell_id=7, genome=[2 0 2 2 2 2 2 4 2 2])]
    assert 0 < cell_lists[2][1].genome[1]   # cell 13 has non-zero CNP in position where potential match has 0
                                            # Genotype(ID=13, Gen=8, cell_id=13, genome=[2 2 2 2 2 2 2 4 2 2])
    distance_provider = SuppliedDistanceProvider(
        ids=[1, 7, 13],
        matrix=np.array([
            [0.0, 3.0, 3.0],
            [3.0, 0.0, 2.0],
            [3.0, 2.0, 0.0],
        ]),
    )
    tt, rt, nj = run_single_test(
        biopsy_generations=[4, 6, 8],
        r_dist=4,
        simulator_with_loaded_tree=sim,
        distance_provider=distance_provider,
        **common_params,
    )
    assert "((7:0.0000)7:3.0000,(13:0.0000)13:3.0000)1;" == to_newick(rt)
    # cell 14 is connected to cell 7, whereas cell 13 not despite dist=0 < r_dist
    # the cell 13 is copied into upper level, and then connected with common father with 7


def test_reconstructor_rule_for_connecting():
    biopsy_set1, biopsy_set2, njbs1, njbs2 = generate_biopsy_sets()
    # biopsy_set1 = [[c1], [c2, c3]] # cannot match 13 to 7
    # biopsy_set2 = [[c4], [c2, c3]] # should match 14 to 5
    # Repeated observations c1/c2 share cell_id 7 and therefore one distance
    # row.
    rt, _, _ = build_evolution_tree(biopsy_set1, dist_matrix_path=data_path("dm", "dm1"), r=2, only_nj=False)
    rt2, _, _ = build_evolution_tree(biopsy_set2, dist_matrix_path=data_path("dm", "dm2"), r=2, only_nj=False)
    assert "((7:0.0000)7:0.0000,(13:0.0000)13:0.0000)None;" == to_newick(rt)
    assert "(7:0.0000,13:0.0000)13;" == to_newick(rt2)
    njt, _, _ = build_evolution_tree(njbs1, dist_matrix_path=data_path("dm", "dm1"), r=2, only_nj=True)
    njt2, _, _ = build_evolution_tree(njbs2, dist_matrix_path=data_path("dm", "dm2"), r=2, only_nj=True)
    assert "(7:0.0000,13:0.0000)None;" == to_newick(njt)
    assert "(13:0.0000,7:0.0000)None;" == to_newick(njt2)


@pytest.mark.parametrize(
    "instr, res", [(""">7
2,0,2,2,2,2,2,4,2,2
>13
2,0,2,2,2,2,2,4,2,2""", 0),
        (""">13
2,0,2,2,2,2,2,4,2,2
>11
2,1,2,2,2,1,2,4,2,2""", 1),
    ]
)
def test_use_cnp2cnp_to_compute_pairwise_distance(instr, res):
    assert res == int(use_cnp2cnp_to_compute_pairwise_distance(instr))


def test_distance_matrix_from_biopsy_rejects_duplicate_distance_ids():
    b, b1, njb, njb1 = generate_biopsy_sets()
    cells = [x for x in flatten(b)]
    with pytest.raises(ValueError, match="Duplicate distance-matrix id 7"):
        distance_matrix_from_biopsy(cells)


def test_reconstructor(show=SHOW_FIGURES):
    b, b1, njb, njb1 = generate_biopsy_sets_small()
    bb = copy.deepcopy(b) # NOTE: (!) needed for r=4 example
    t, l, _ = build_evolution_tree(b, dist_matrix_path=data_path("dm", "distance_matrix.txt"), r=2)
    assert to_newick(t) == "((4:0.0000)4:1.7500,(1:0.5000,(3:2.0000)2:0.5000)None:1.7500)None;" #3->2
    if show: visualize_tree_plotly(t, l)
    t, l, _ = build_evolution_tree(b1, dist_matrix_path=data_path("dm", "distance_matrix.txt"), r=2)
    assert to_newick(t) == "((4:0.0000)4:1.7500,((3:1.0000)1:0.5000,2:0.5000)None:1.7500)None;" #3->1
    if show: visualize_tree_plotly(t, l)
    t, l, _ = build_evolution_tree(bb, dist_matrix_path=data_path("dm", "distance_matrix.txt"), r=4)
    assert to_newick(t) == "(1:0.5000,(3:2.0000,4:4.0000)2:0.5000)None;" # 3->2, 4->2
    if show: visualize_tree_plotly(t, l)
    t, l, _ = build_evolution_tree(njb, dist_matrix_path=data_path("dm", "distance_matrix.txt"), r=4, only_nj=True)
    assert to_newick(t) == "((1:0.2500,2:0.7500)None:0.1250,(3:0.7500,4:3.2500)None:0.1250)None;"
    if show: visualize_tree_plotly(t, l)
    t, l, _ = build_evolution_tree(njb1, dist_matrix_path=data_path("dm", "distance_matrix.txt"), r=1, only_nj=True)
    assert to_newick(t) == "((1:0.2500,2:0.7500)None:0.1250,(3:0.7500,4:3.2500)None:0.1250)None;" #same as previous NJ
    if show: visualize_tree_plotly(t, l)


# def test_reconstructor_full(show=SHOW_FIGURES):
#     b, b1, njb, njb1 = generate_biopsy_sets_small()
#     bb = copy.deepcopy(b) # NOTE: (!) needed for r=4 example
#     t, l, _ = build_evolution_tree(b, dist_matrix_path="data/dm/distance_matrix.txt", r=2,
#                                    neighbor_joining=neighbor_joining_full)
#     assert to_newick(t) == "(((3:2.0000)2:0.0000,1:1.0000)2:0.0000,(4:0.0000)4:4.0000)2;"
#     if show: visualize_tree_plotly(t, l)
#     t, l, _ = build_evolution_tree(b1, dist_matrix_path="data/dm/distance_matrix.txt", r=2,
#                                    neighbor_joining=neighbor_joining_full)
#     assert to_newick(t) == "(((3:1.0000)1:0.0000,2:1.0000)1:0.0000,(4:0.0000)4:4.0000)1;"
#     if show: visualize_tree_plotly(t, l)
#     t, l, _ = build_evolution_tree(bb, dist_matrix_path="data/dm/distance_matrix.txt", r=4,
#                                    neighbor_joining=neighbor_joining_full)
#     assert to_newick(t) == "((3:2.0000,4:4.0000)2:0.0000,1:1.0000)2;"
#     if show: visualize_tree_plotly(t, l)
#     t, l, _ = build_evolution_tree(njb, dist_matrix_path="data/dm/distance_matrix.txt", r=4, only_nj=True,
#                                    neighbor_joining=neighbor_joining_full)
#     assert to_newick(t) == "(((2:0.0000,1:1.0000)2:0.0000,3:2.0000)2:0.0000,4:4.0000)2;"
#     if show: visualize_tree_plotly(t, l)
#     t, l, _ = build_evolution_tree(njb1, dist_matrix_path="data/dm/distance_matrix.txt", r=1, only_nj=True,
#                                    neighbor_joining=neighbor_joining_full)
#     assert to_newick(t) == "(((1:0.0000,2:1.0000)1:0.0000,3:1.0000)1:0.0000,4:4.0000)1;"
#     if show: visualize_tree_plotly(t, l)


# def test_reconstructor_njfull():
#     D = np.array([
#         [0, 1, 1],
#         [1, 0, 4],
#         [1, 4, 0]
#     ])
#     cells = [Genotype([2, 2, 1], 1), Genotype([1, 1, 1], 2), Genotype([2, 1, 1], 3)]
#     c3 = copy.deepcopy(cells)
#     max_id = 3
#     t1, l, _ = neighbor_joining_full(D, cells, max_id, existing_tree=None)
#     assert to_newick(t1) == "((1:0.0000,2:1.0000)1:0.0000,3:1.0000)1;"
#     D = np.array([
#         [0, 1, 3, 8],
#         [1, 0, 7, 7],
#         [3, 7, 0, 2],
#         [8, 7, 2, 0]
#     ])
#     cells = [Genotype([2, 2, 1], 1), Genotype([1, 1, 1], 2),
#              Genotype([2, 1, 1], 3), Genotype([2, 1, 7], 4)]
#     c4 = copy.deepcopy(cells)
#     max_id = 4
#     t2, l, _ = neighbor_joining_full(D, cells, max_id, existing_tree=None)
#     assert to_newick(t2) == "((1:0.0000,2:1.0000)1:0.0000,(3:0.0000,4:2.0000)3:3.0000)1;"
#
#     t3, l, _ = build_evolution_tree([c3], dist_matrix_path="data/dm/dm3", only_nj=True,
#                                     neighbor_joining=neighbor_joining_full)
#     assert to_newick(t1) == to_newick(t3)
#
#     t4, l, _ = build_evolution_tree([c3], dist_matrix_path="data/dm/dm3",
#                                     neighbor_joining=neighbor_joining_full)
#     assert to_newick(t1) == to_newick(t4)
#
#     t5, l, _ = build_evolution_tree([c4], dist_matrix_path="data/dm/dm4", only_nj=True,
#                                     neighbor_joining=neighbor_joining_full)
#     assert to_newick(t2) == to_newick(t5)
#
#     t6, l, _ = build_evolution_tree([c4], dist_matrix_path="data/dm/dm4",
#                                     neighbor_joining=neighbor_joining_full)
#     assert to_newick(t2) == to_newick(t6)


# @pytest.mark.parametrize("fun", [neighbor_joining_full,
#                                  neighbor_joining_full_cps,
#                                  neighbor_joining_hybrid,
#                                  neighbor_joining_hybrid_inverse_centrality
#                                  ])
# def test_reconstructor_plausability(fun):
#     a, b, c = run_single_test(
#         seed=95,
#         config="data/config_for_pic.json",
#         bedfile="data/pic.csv",
#         biopsy_size_scalable=0.5,
#         biopsy_generations=[4, 6, 8],
#         r_dist=4,
#         write_newick=True,
#         reconstruction_algorithm=fun,
#     )
