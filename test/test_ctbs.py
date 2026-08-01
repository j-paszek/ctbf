import copy

import numpy as np
import pandas as pd
import pytest
from networkx.utils.misc import flatten
import atexit
import os

from ctbs import (
    SuppliedDistanceProvider,
    run_single_test,
    use_cnp2cnp_to_compute_pairwise_distance,
    distance_matrix_from_biopsy,
)
from ctbs_utils import to_newick, vizualize_nx_tree, vizualize_from_newick

import json
import networkx as nx
from networkx.readwrite import json_graph

import ctbs
from evaluator_full import evaluate_4
from reconstructor import (build_evolution_tree, visualize_tree_plotly,
                           # neighbor_joining_full, neighbor_joining_full_cps, \
    # neighbor_joining_hybrid, neighbor_joining_hybrid_inverse_centrality, neighbor_joining_adaptive_centrality, \
    neighbor_joining_adaptive_centrality_nonlinear)
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


def tree_to_json(tree):
    data = json_graph.node_link_data(tree)
    with open(data_path("tree_samples", str(tree) + ".json"), "w") as f:
        json.dump(data, f, indent=2)


def tree_from_json(tr_id):
    with open(data_path("tree_samples", str(tr_id) + ".json")) as f:
        data_loaded = json.load(f)
    return json_graph.node_link_graph(data_loaded, directed=True)


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


def test_no_nj_in_reconstruction():
    # Biopsy: [5]                   level 1
    # Biopsy: [15, 3]               level 2
    # Biopsy: [3, 22, 15, 23]       level 3
    # Edge case - top level is of size one; all elements from next level are in r_dist range
    # reconstructor in build_evolution_tree pass to neighbor_joining already completed tree
    # Reconstructed tree is ((3,22)3,(15,23)15)5
    run_single_test(config=data_path("config_for_pic.json"), bedfile=data_path("pic.csv"),
                    seed=582, biopsy_size_scalable=0.5,
                    biopsy_generations=[4, 6, 8], r_dist=4, clear_cnps=False)


# def test_no_nj_in_reconstruction_full():
#     # Biopsy: [5]                   level 1
#     # Biopsy: [15, 3]               level 2
#     # Biopsy: [3, 22, 15, 23]       level 3
#     # Edge case - top level is of size one; all elements from next level are in r_dist range
#     # reconstructor in build_evolution_tree pass to neighbor_joining already completed tree
#     # Reconstructed tree is ((3,22)3,(15,23)15)5
#     run_single_test(config="data/config_for_pic.json", bedfile="data/pic.csv",
#                     seed=582, biopsy_size_scalable=0.5,
#                     biopsy_generations=[4, 6, 8], r_dist=4, clear_cnps=False,
#                     reconstruction_algorithm=neighbor_joining_full)


def test_scalable_biopsy_samples_one_cell_from_nonempty_generation():
    sim = get_sim_from_tree(689)
    biopsy = sim.perform_biopsy(biopsy_size_scalable=0.5, generation=4, seed=689)
    assert [(cell.node_id, cell.cell_id) for cell in biopsy] == [(4, 1)]


def test_scalable_biopsy_keeps_empty_generation_empty():
    sim = get_sim_from_tree(689)
    biopsy = sim.perform_biopsy(biopsy_size_scalable=0.5, generation=99, seed=689)
    assert biopsy == []


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


def test_empty_biopsy_ctbf(): #capsys
    sim = get_sim_from_tree(689)
    # tt, rt, nj = run_single_test(seed=689, biopsy_size_scalable=0.5, biopsy_generatons=[4, 6, 8],
    #                          r_dist=4, simlulator_with_loaded_tree=sim)
    # captured = capsys.readouterr()
    # assert "Biopsy sample from generation  4  has no cells. Skipping." in captured.out


def test_reconstructor_no_connecting_within_distance():
    sim = get_sim_from_tree(689)
    bg = [4, 6, 8]
    common_params = {
        "seed": 689,
        "biopsy_size_scalable": 0.5
    }
    cell_lists = []
    for b_gen in bg:
        biopsy = sim.perform_biopsy(generation=b_gen, **common_params)
        if biopsy:
            cell_lists.append(biopsy)
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
    rt, _, _ = build_evolution_tree(biopsy_set1, dist_matrix_path=data_path("dm", "dm1"), r=2, only_nj=False)
    rt2, _, _ = build_evolution_tree(biopsy_set2, dist_matrix_path=data_path("dm", "dm2"), r=2, only_nj=False)
    assert "((7:0.0000)7:0.0000,(13:0.0000)13:0.0000)None;" == to_newick(rt)
    assert "(7:0.0000,13:0.0000)13;" == to_newick(rt2)
    njt, _, _ = build_evolution_tree(njbs1, dist_matrix_path=data_path("dm", "dm1"), r=2, only_nj=True)
    njt2, _, _ = build_evolution_tree(njbs2, dist_matrix_path=data_path("dm", "dm2"), r=2, only_nj=True)
    assert "(7:0.0000,13:0.0000)None;" == to_newick(njt)
    assert "(13:0.0000,7:0.0000)None;" == to_newick(njt2)


def test_simulator_legacy():
    tt, rt, nj = run_single_test(seed=689, config=data_path("config_for_pic.json"), bedfile=data_path("pic.csv"),
                    biopsy_size_scalable=0.5, biopsy_generations=[4, 6, 8], r_dist=4)
    t689 = tree_from_json(689)
    assert not to_newick(t689) == to_newick(tt)
    # we want it to fail to produce a tree, where for some nodes A and B; A is parent of B;
    # and for some cnp position i, we have in node A cnp(i)=0 and for node B cnp(i)<>0


# def test_no_skipping_cells_between_biopsies():
#     a,b,c = run_single_test(seed=35, config="data/config_for_pic.json", bedfile="data/pic.csv",
#                     biopsy_size_scalable=0.5, biopsy_generations=[4, 6, 8], r_dist=4, write_newick=True,
#                     reconstruction_algorithm=neighbor_joining_full)
#     assert to_newick(a) == "((((((((6)6,(58:2,39)39:2)6,((26)26)26:1)6)6)6:1,(((((41)41:1)18)18:1)12:1,((((1)1)1,((63:1)29,(64:2,44,66:2,67:1)44:1)29:1)1,(((45)45:1)20,((31)31)31:1)20:1)1)1)1)1:1,(((((((32,71:1)32)32:2,((8)8)8)8)8)8:1,(((((49)49:1,22)22)22:1,(((74:2)15,(52)52:1)15)15)15:1)0)0,((((((76:1,53)53:1,(24)24)24,((55)55:1,(56,81:1)56:1)37:1)24:1)16:1)5)5:1)0)0;"
#     assert to_newick(b) == "(((24:1.0000,76:3.0000,53:2.0000)37:2.0000)16:0.0000,(((67:2.0000,26:2.0000,71:4.0000)31:2.0000,(63:1.0000,39:1.0000)29:1.0000,(41:1.0000)18:2.0000,(32:3.0000,49:1.0000)15:2.0000,20:1.0000,(1:0.0000)1:0.0000)1:0.0000,6:1.0000)1:2.0000)16;"
#     # before random; choice by id  "((((67:2.0000,26:2.0000,71:4.0000)31:2.0000,(63:1.0000,39:1.0000)29:1.0000,(41:1.0000)18:2.0000,(32:3.0000,49:1.0000)15:2.0000,20:1.0000,(1:0.0000)1:0.0000)1:0.0000,6:1.0000)1:0.0000,((24:1.0000,76:3.0000,53:2.0000)37:2.0000)16:2.0000)1;")
#     assert to_newick(c) == "((((53:0.0000,76:1.0000)53:0.0000,37:2.0000)53:0.0000,((26:0.0000,(((((((((1:0.0000,6:1.0000)1:0.0000,29:1.0000)1:0.0000,20:1.0000)1:0.0000,(16:0.0000,24:1.0000)16:2.0000)1:0.0000,31:2.0000)1:0.0000,(18:0.0000,41:1.0000)18:2.0000)1:0.0000,(15:0.0000,49:1.0000)15:2.0000)1:0.0000,63:2.0000)1:0.0000,67:2.0000)1:2.0000)26:0.0000,39:2.0000)26:2.0000)53:0.0000,(32:0.0000,71:1.0000)32:3.0000)53;"
#     # x = "((32:0.0000,71:1.0000)32:0.0000,(((53:0.0000,76:1.0000)53:0.0000,37:2.0000)53:0.0000,((26:0.0000,(((((((((1:0.0000,6:1.0000)1:0.0000,29:1.0000)1:0.0000,20:1.0000)1:0.0000,(16:0.0000,24:1.0000)16:2.0000)1:0.0000,31:2.0000)1:0.0000,(18:0.0000,41:1.0000)18:2.0000)1:0.0000,(15:0.0000,49:1.0000)15:2.0000)1:0.0000,63:2.0000)1:0.0000,67:2.0000)1:2.0000)26:0.0000,39:2.0000)26:2.0000)53:3.0000)32;"
#     # y = "(((((((((((((1:0.0000,6:1.0000)1:0.0000,29:1.0000)1:0.0000,20:1.0000)1:0.0000,(16:0.0000,24:1.0000)16:2.0000)1:0.0000,31:2.0000)1:0.0000,(18:0.0000,41:1.0000)18:2.0000)1:0.0000,(15:0.0000,49:1.0000)15:2.0000)1:0.0000,63:2.0000)1:0.0000,67:2.0000)1:0.0000,26:2.0000)1:0.0000,39:2.0000)1:0.0000,(37:0.0000,(53:0.0000,76:1.0000)53:2.0000)37:3.0000)1:0.0000,(32:0.0000,71:1.0000)32:3.0000)1;"
#     # vizualize_from_newick(x)
#     # vizualize_nx_tree(c) # 1-1 instead of 1-29 on the lowest levels


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


def test_reconstructor_with_parallel():
    simt, rt, njt = run_single_test(config=data_path("config100.json"), bedfile=None, seed=777,
                    biopsy_size=2, biopsy_size_scalable=None, biopsy_generations=[5, 7, 9], r_dist=4,
                    visualize=False, time_collector=None, clear_cnps=False, compare_dm=False,
                    write_newick=False, simulator_with_loaded_tree=None, parallel=False)
    simt1, rt1, njt1 = run_single_test(config=data_path("config100.json"), bedfile=None, seed=777,
                                    biopsy_size=2, biopsy_size_scalable=None, biopsy_generations=[5, 7, 9], r_dist=4,
                                    visualize=False, time_collector=None, clear_cnps=False, compare_dm=False,
                                    write_newick=False, simulator_with_loaded_tree=None, parallel=True)
    assert to_newick(simt) == to_newick(simt1)
    assert to_newick(njt) == to_newick(njt1)
    assert to_newick(rt) == to_newick(rt1)



def test_empty_biopsy():
    # Edge case - simulator tries to do a duplication on a position i s.t. cnp(i) = 0
    # TODO: Shall we enable appearance of new copies of cnps
    # Edge case - cell_7 cnp =[2,0,2,2,2,2,2,4,2,2] cell_13 cnp=[2,2,2,2,2,2,2,4,2,2]
    # FIXME: cnp2cnp(cell_7, cell_13) = 0; they have the assumption that when cnp(i) reaches 0 it stays 0
    # that means cnp2cnp runs as cnp2cnp([2,  2,2,2,2,2,4,2,2], [2,  2,2,2,2,2,4,2,2])
    # we omit second position even in cell_13; in this edge case cell_7 and cell_13 becomes identical
    run_single_test(seed=689, config=data_path("config_for_pic.json"), bedfile=data_path("pic.csv"),
                    biopsy_size_scalable=0.5, biopsy_generations=[4, 6, 8], r_dist=4)


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


df = pd.read_csv(os.path.join(TEST_DATA_DIR, "f1results.csv"), delimiter="\t")
def float_to_str_comma(x, digits=3):
    """
    Convert a float to string with comma as decimal separator and fixed number of digits.
    Example: 0.056644 -> '0,057'
    """
    rounded = round(x, digits)
    int_part = int(rounded)

    # check if number is integer after rounding
    if rounded == int_part:
        return f"{int_part},0"
    else:
        s = f"{rounded:.{digits}f}".rstrip("0")
        if s.endswith("."):
            s += "0"
        return s.replace(".", ",")

@pytest.mark.skip(reason="One time test before code integration.")
@pytest.mark.parametrize("seed,t,tr,tnj,tp,fp,fn,f1,i,p,r",
                         [(row["seed"], row["T"], row["Trec"], row["Tnj"],
                           row["TP"], row["FP"], row["FN"], row["F1"], row["IoU"],
                           row["precision"], row["recall"]) for _, row in df.iterrows()])
def test_ete3_to_nx(seed, t, tr, tnj, tp, fp, fn, f1, i, p, r):
    a,b,c = run_single_test(seed=seed, config=data_path("config_for_pic.json"), bedfile=data_path("pic.csv"),
                    biopsy_size_scalable=0.5, biopsy_generations=[4, 6, 8], r_dist=4, write_newick=True,
                    reconstruction_algorithm=neighbor_joining_full)
    # Checking trees
    assert to_newick(a) == t
    if "-" in tnj:
        assert to_newick(b) == tr
        out = evaluate_4(a, b)
        tp_exp = out["ancestors_multiset"]["TP"]
        assert tp == tp_exp
        fp_exp = out["ancestors_multiset"]["FP"]
        assert fp == fp_exp
        fn_exp = out["ancestors_multiset"]["FN"]
        assert fn == fn_exp
        f1_exp = out["ancestors_multiset"]["F1"]
        # assert float(f1.replace(",", ".")) == pytest.approx(f1_exp, 1e-3)
        assert f1 == float_to_str_comma(f1_exp, 3)
        i_exp = out["ancestors_multiset"]["IoU"]
        assert i == float_to_str_comma(i_exp, 3)
        p_exp = out["ancestors_multiset"]["precision"]
        assert p == float_to_str_comma(p_exp, 3)
        r_exp = out["ancestors_multiset"]["recall"]
        assert r == float_to_str_comma(r_exp, 3)
    else:
        assert to_newick(c) == tnj
        out = evaluate_4(a, c)
        tp_exp = out["ancestors_multiset"]["TP"]
        assert tp == tp_exp
        fp_exp = out["ancestors_multiset"]["FP"]
        assert fp == fp_exp
        fn_exp = out["ancestors_multiset"]["FN"]
        assert fn == fn_exp
        f1_exp = out["ancestors_multiset"]["F1"]
        # assert float(f1.replace(",", ".")) == pytest.approx(f1_exp, 1e-3)
        assert f1 == float_to_str_comma(f1_exp, 3)
        i_exp = out["ancestors_multiset"]["IoU"]
        assert i == float_to_str_comma(i_exp, 3)
        p_exp = out["ancestors_multiset"]["precision"]
        assert p == float_to_str_comma(p_exp, 3)
        r_exp = out["ancestors_multiset"]["recall"]
        assert r == float_to_str_comma(r_exp, 3)

df = pd.read_csv(os.path.join(TEST_DATA_DIR, "f1results.csv"), delimiter="\t")
results_store = {"ancestors_multiset_precision_failures": [], "ancestors_multiset_f1_failures": [],
                 "ancestors_unique_precision_failures": [], "ancestors_unique_f1_failures": []}
@pytest.fixture
def failures():
    return results_store

rec_output = {"seed": [], "1-precision": [], "1-f1": [], "2-precision": [], "2-f1": []}
@pytest.fixture
def rec_report():
    return rec_output

nj_output = {"seed": [], "1-precision": [], "1-f1": [], "2-precision": [], "2-f1": []}
@pytest.fixture
def nj_report():
    return nj_output

def provide_summary(rec, nj, mode, seed, failures, rec_report, nj_report):
    p1 = rec[mode]["precision"]
    p2 = nj[mode]["precision"]
    f1 = rec[mode]["F1"]
    f2 = nj[mode]["F1"]
    if mode == "ancestors_multiset":
        rec_report["seed"].append(seed)
        rec_report["1-precision"].append(p1)
        rec_report["1-f1"].append(f1)
    else:
        rec_report["2-precision"].append(p1)
        rec_report["2-f1"].append(f1)
    if mode == "ancestors_multiset":
        nj_report["seed"].append(seed)
        nj_report["1-precision"].append(p2)
        nj_report["1-f1"].append(f2)
    else:
        nj_report["2-precision"].append(p2)
        nj_report["2-f1"].append(f2)
    print("Mode ", mode)
    print(f"Precision: {p1} vs: {p2}")
    print(f"F1: {f1} vs: {f2}")
    if p1 < p2:
        name = mode + "_precision_failures"
        failures[name].append(seed)
    if f1 < f2:
        name = mode + "_f1_failures"
        failures[name].append(seed)


@pytest.mark.parametrize("seed", df["seed"].unique().tolist())
def test_evaluate4(seed, failures, rec_report, nj_report):
    print(f"\nTesting seed: {seed}")
    a, b, c = run_single_test(seed=seed, config=data_path("config_for_pic.json"), bedfile=data_path("pic.csv"),
                              biopsy_size_scalable=0.5, biopsy_generations=[4, 6, 8], r_dist=4, write_newick=True,
                              reconstruction_algorithm=neighbor_joining_adaptive_centrality_nonlinear)
    rec = evaluate_4(a, b)
    nj = evaluate_4(a, c)
    for mode in ["ancestors_multiset", "ancestors_unique"]:
        provide_summary(rec, nj, mode, seed, failures, rec_report, nj_report)


# register a function that will run even under PyCharm
def save_results():
    print("\n\nSummary")
    print(results_store)
    all_seeds = sorted({s for v in results_store.values() for s in v})

    # initialize df
    df = pd.DataFrame({"seed": all_seeds})

    # mark presence (1 if seed in that list)
    for key, seeds in results_store.items():
        df[key] = df["seed"].apply(lambda x: 1 if x in seeds else 0)

    # --- Optional: add a summary column ---
    df["total_failures"] = df.drop(columns=["seed"]).sum(axis=1)

    df.to_csv('out.csv', index=False)

    print(rec_output)
    df1 = pd.DataFrame(rec_output)
    df1.to_csv('rec.csv', index=False)

    print(nj_output)
    df2 = pd.DataFrame(rec_output)
    df2.to_csv('nj.csv', index=False)

atexit.register(save_results)
