from ctbs import run_single_test


def test_run_single_test():
    # Biopsy: [5]                   level 1
    # Biopsy: [15, 3]               level 2
    # Biopsy: [3, 22, 15, 23]       level 3
    # Edge case - top level is of size one; all elements from next level are in r_dist range
    # reconstructor in build_evolution_tree pass to neighbor_joining already completed tree
    # Reconstructed tree is ((3,22)3,(15,23)15)5
    run_single_test(config="data/config_for_pic.json", bedfile="data/pic.csv",
                    seed=582, biopsy_size_scalable=0.5,
                    biopsy_generatons=[4, 6, 8], r_dist=4, clear_cnps=False)
