import copy

import numpy as np
import pytest
from networkx.utils.misc import flatten

from ctbs import run_single_test, use_cnp2cnp_to_compute_pairwise_distance, distance_matrix_from_biopsy
from ctbs_utils import to_newick

import json
import networkx as nx
from networkx.readwrite import json_graph

from reconstructor import build_evolution_tree, visualize_tree_plotly, neighbor_joining_full
from simulator import CancerCellEvolutionSimulator, Genotype

SHOW_FIGURES = False

def get_sim_from_tree(tr):
    tree = tree_from_json(tr)
    sim = CancerCellEvolutionSimulator.from_tree(tree)
    return sim


def tree_to_json(tree):
    data = json_graph.node_link_data(tree)
    with open("data/tree_samples/"+str(tree)+".json", "w") as f:
        json.dump(data, f, indent=2)


def tree_from_json(tr_id):
    with open("data/tree_samples/"+str(tr_id)+".json") as f:
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
    run_single_test(config="data/config_for_pic.json", bedfile="data/pic.csv",
                    seed=582, biopsy_size_scalable=0.5,
                    biopsy_generations=[4, 6, 8], r_dist=4, clear_cnps=False)


def test_no_nj_in_reconstruction_full():
    # Biopsy: [5]                   level 1
    # Biopsy: [15, 3]               level 2
    # Biopsy: [3, 22, 15, 23]       level 3
    # Edge case - top level is of size one; all elements from next level are in r_dist range
    # reconstructor in build_evolution_tree pass to neighbor_joining already completed tree
    # Reconstructed tree is ((3,22)3,(15,23)15)5
    run_single_test(config="data/config_for_pic.json", bedfile="data/pic.csv",
                    seed=582, biopsy_size_scalable=0.5,
                    biopsy_generations=[4, 6, 8], r_dist=4, clear_cnps=False,
                    reconstruction_algorithm=neighbor_joining_full)


def test_empty_biopsy_simulator():
    sim = get_sim_from_tree(689)
    biopsy = sim.perform_biopsy(biopsy_size_scalable=0.5, generation=4, seed=689)
    assert biopsy == []


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
    assert [[7], [7, 13]] == [[c.cell_id for c in cl] for cl in cell_lists]
    # empty biopsy from generation 4 is not included; same genotype in two biopsies
    assert [[7], [14, 13]] == [[c.node_id for c in cl] for cl in cell_lists]
    # cells 7 and 14 share the same genotype
    assert 0 == cell_lists[0][0].genome[1]  # cell 7 (oldest) has 0 in CNP (for i=1, cnp(i)=0)
                                            # [Genotype(ID=7, Gen=6, cell_id=7, genome=[2 0 2 2 2 2 2 4 2 2])]
    assert 0 < cell_lists[1][1].genome[1]   # cell 13 has non-zero CNP in position where potential match has 0
                                            # Genotype(ID=13, Gen=8, cell_id=13, genome=[2 2 2 2 2 2 2 4 2 2])
    tt, rt, nj = run_single_test(biopsy_generations=[4, 6, 8], r_dist=4, simulator_with_loaded_tree=sim, **common_params)
    assert "((7:0.0000)7:0.0000,(13:0.0000)13:0.0000)None;" == to_newick(rt)
    # cell 14 is connected to cell 7, whereas cell 13 not despite dist=0 < r_dist
    # the cell 13 is copied into upper level, and then connected with common father with 7


def test_reconstructor_rule_for_connecting():
    biopsy_set1, biopsy_set2, njbs1, njbs2 = generate_biopsy_sets()
    # biopsy_set1 = [[c1], [c2, c3]] # cannot match 13 to 7
    # biopsy_set2 = [[c4], [c2, c3]] # should match 14 to 5
    rt, _, _ = build_evolution_tree(biopsy_set1, "data/dm/dm1", r=2, only_nj=False)
    rt2, _, _ = build_evolution_tree(biopsy_set2, "data/dm/dm2", r=2, only_nj=False)
    assert "((7:0.0000)7:0.0000,(13:0.0000)13:0.0000)None;" == to_newick(rt)
    assert "(7:0.0000,13:0.0000)13;" == to_newick(rt2)
    njt, _, _ = build_evolution_tree(njbs1, "data/dm/dm1", r=2, only_nj=True)
    njt2, _, _ = build_evolution_tree(njbs2, "data/dm/dm2", r=2, only_nj=True)
    assert "(7:0.0000,13:0.0000)None;" == to_newick(njt)
    assert "(13:0.0000,7:0.0000)None;" == to_newick(njt2)


def test_simulator_legacy():
    tt, rt, nj = run_single_test(seed=689, config="data/config_for_pic.json", bedfile="data/pic.csv",
                    biopsy_size_scalable=0.5, biopsy_generations=[4, 6, 8], r_dist=4)
    t689 = tree_from_json(689)
    assert not to_newick(t689) == to_newick(tt)
    # we want it to fail to produce a tree, where for some nodes A and B; A is parent of B;
    # and for some cnp position i, we have in node A cnp(i)=0 and for node B cnp(i)<>0


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


def test_distance_matrix_from_biopsy():
    b, b1, njb, njb1 = generate_biopsy_sets()
    cells = [x for x in flatten(b)]
    a, b = distance_matrix_from_biopsy(cells)
    print(a, b)
    assert np.array_equal(a, [7, 7, 13])
    assert np.array_equal(b, [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])


def test_reconstructor_with_parallel():
    simt, rt, njt = run_single_test(config="data/config100.json", bedfile=None, seed=777,
                    biopsy_size=2, biopsy_size_scalable=None, biopsy_generations=[5, 7, 9], r_dist=4,
                    visualize=False, time_collector=None, clear_cnps=False, compare_dm=False,
                    write_newick=False, simulator_with_loaded_tree=None, parallel=False)
    simt1, rt1, njt1 = run_single_test(config="data/config100.json", bedfile=None, seed=777,
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
    run_single_test(seed=689, config="data/config_for_pic.json", bedfile="data/pic.csv",
                    biopsy_size_scalable=0.5, biopsy_generations=[4, 6, 8], r_dist=4)


def test_reconstructor(show=SHOW_FIGURES):
    b, b1, njb, njb1 = generate_biopsy_sets_small()
    bb = copy.deepcopy(b) # NOTE: (!) needed for r=4 example
    t, l, _ = build_evolution_tree(b, "data/dm/distance_matrix.txt", r=2)
    assert to_newick(t) == "((4:0.0000)4:1.7500,(1:0.5000,(3:2.0000)2:0.5000)None:1.7500)None;" #3->2
    if show: visualize_tree_plotly(t, l)
    t, l, _ = build_evolution_tree(b1, "data/dm/distance_matrix.txt", r=2)
    assert to_newick(t) == "((4:0.0000)4:1.7500,((3:1.0000)1:0.5000,2:0.5000)None:1.7500)None;" #3->1
    if show: visualize_tree_plotly(t, l)
    t, l, _ = build_evolution_tree(bb, "data/dm/distance_matrix.txt", r=4)
    assert to_newick(t) == "(1:0.5000,(3:2.0000,4:4.0000)2:0.5000)None;" # 3->2, 4->2
    if show: visualize_tree_plotly(t, l)
    t, l, _ = build_evolution_tree(njb, "data/dm/distance_matrix.txt", r=4, only_nj=True)
    assert to_newick(t) == "((1:0.2500,2:0.7500)None:0.1250,(3:0.7500,4:3.2500)None:0.1250)None;"
    if show: visualize_tree_plotly(t, l)
    t, l, _ = build_evolution_tree(njb1, "data/dm/distance_matrix.txt", r=1, only_nj=True)
    assert to_newick(t) == "((1:0.2500,2:0.7500)None:0.1250,(3:0.7500,4:3.2500)None:0.1250)None;" #same as previous NJ
    if show: visualize_tree_plotly(t, l)


def test_reconstructor_full(show=SHOW_FIGURES):
    b, b1, njb, njb1 = generate_biopsy_sets_small()
    bb = copy.deepcopy(b) # NOTE: (!) needed for r=4 example
    t, l, _ = build_evolution_tree(b, "data/dm/distance_matrix.txt", r=2,
                                   neighbor_joining=neighbor_joining_full)
    assert to_newick(t) == "((1:0.0000,(3:2.0000)2:1.0000)1:0.0000,(4:0.0000)4:4.0000)1;"
    if show: visualize_tree_plotly(t, l)
    t, l, _ = build_evolution_tree(b1, "data/dm/distance_matrix.txt", r=2,
                                   neighbor_joining=neighbor_joining_full)
    assert to_newick(t) == "(((3:1.0000)1:0.0000,2:1.0000)1:0.0000,(4:0.0000)4:4.0000)1;"
    if show: visualize_tree_plotly(t, l)
    t, l, _ = build_evolution_tree(bb, "data/dm/distance_matrix.txt", r=4,
                                   neighbor_joining=neighbor_joining_full)
    assert to_newick(t) == "(1:0.0000,(3:2.0000,4:4.0000)2:1.0000)1;"
    if show: visualize_tree_plotly(t, l)
    t, l, _ = build_evolution_tree(njb, "data/dm/distance_matrix.txt", r=4, only_nj=True,
                                   neighbor_joining=neighbor_joining_full)
    assert to_newick(t) == "(((1:0.0000,2:1.0000)1:0.0000,3:1.0000)1:0.0000,4:4.0000)1;"
    if show: visualize_tree_plotly(t, l)
    t, l, _ = build_evolution_tree(njb1, "data/dm/distance_matrix.txt", r=1, only_nj=True,
                                   neighbor_joining=neighbor_joining_full)
    assert to_newick(t) == "(((1:0.0000,2:1.0000)1:0.0000,3:1.0000)1:0.0000,4:4.0000)1;"
    if show: visualize_tree_plotly(t, l)


def test_reconstructor_njfull():
    D = np.array([
        [0, 1, 1],
        [1, 0, 4],
        [1, 4, 0]
    ])
    cells = [Genotype([2, 2, 1], 1), Genotype([1, 1, 1], 2), Genotype([2, 1, 1], 3)]
    c3 = copy.deepcopy(cells)
    max_id = 3
    t1, l, _ = neighbor_joining_full(D, cells, max_id, existing_tree=None)
    assert to_newick(t1) == "((1:0.0000,2:1.0000)1:0.0000,3:1.0000)1;"
    D = np.array([
        [0, 1, 3, 8],
        [1, 0, 7, 7],
        [3, 7, 0, 2],
        [8, 7, 2, 0]
    ])
    cells = [Genotype([2, 2, 1], 1), Genotype([1, 1, 1], 2),
             Genotype([2, 1, 1], 3), Genotype([2, 1, 7], 4)]
    c4 = copy.deepcopy(cells)
    max_id = 4
    t2, l, _ = neighbor_joining_full(D, cells, max_id, existing_tree=None)
    assert to_newick(t2) == "((1:0.0000,2:1.0000)1:0.0000,(3:0.0000,4:2.0000)3:3.0000)1;"

    t3, l, _ = build_evolution_tree([c3], dist_matrix_path="data/dm/dm3", only_nj=True,
                                    neighbor_joining=neighbor_joining_full)
    assert to_newick(t1) == to_newick(t3)

    t4, l, _ = build_evolution_tree([c3], dist_matrix_path="data/dm/dm3",
                                    neighbor_joining=neighbor_joining_full)
    assert to_newick(t1) == to_newick(t4)

    t5, l, _ = build_evolution_tree([c4], dist_matrix_path="data/dm/dm4", only_nj=True,
                                    neighbor_joining=neighbor_joining_full)
    assert to_newick(t2) == to_newick(t5)

    t6, l, _ = build_evolution_tree([c4], dist_matrix_path="data/dm/dm4",
                                    neighbor_joining=neighbor_joining_full)
    assert to_newick(t2) == to_newick(t6)
