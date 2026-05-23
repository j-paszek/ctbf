import shutil
import subprocess
import os.path
import sys
import tempfile
import time
import json
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor

from ctbf_constraints import MIN_TOTAL_BIOPSY_CELLS
from simulator import CancerCellEvolutionSimulator, Genotype
from reconstructor import build_evolution_tree, resolve_biopsy_guided_config, visualize_tree_plotly
from reconstructor_registry import get_algorithm_map, resolve_reconstruction_algorithm
from evaluator import grf_tree
from evaluator_full import evaluate_4, named_label
from ctbs_utils import to_newick, vizualize_nx_tree, get_biopsy_nodes_ids

DEFAULT_CTBS_CONFIG = {
    "IN_FILE_NAME": "biopsy.txt",
    "OUT_FILE_NAME": "cnp_distance_matrix.txt",
    "SIM_DM": "sim_dm.txt",
    "cnp2cnp_FOLDER": "/Users/voronwe/Work/PyCharmProjects/cnp2cnp/examples",
    "cnp2cnp_FILE": "/Users/voronwe/Work/PyCharmProjects/cnp2cnp/cnp2cnp.py",
    "TRUE_TREE_ROOT_ID": 0,
    "RUN_SINGLE_TEST": {
        "seed": 2,
        "config": "test/data/config_for_pic.json",
        "bedfile": "test/data/pic.csv",
        "biopsy_size_scalable": 0.5,
        "biopsy_generations": [3, 5],
        "r_dist": 4,
        "write_newick": True,
        "visualize": True,
        "reconstruction_algorithm": "neighbor_joining_hybrid_anticentral_adaptive_v3",
        "biopsy_guided_strategy": None,
    },
}
CTBS_CONFIG_PATH = Path(__file__).with_name("ctbs_config.json")

RECONSTRUCTION_ALGORITHMS = get_algorithm_map()


@dataclass(frozen=True)
class CtbsRuntimeConfig:
    in_file_name: str
    out_file_name: str
    sim_dm: str
    cnp2cnp_folder: str
    cnp2cnp_file: str
    true_tree_root_id: int
    run_single_test: dict

    @classmethod
    def from_mapping(cls, config):
        return cls(
            in_file_name=config["IN_FILE_NAME"],
            out_file_name=config["OUT_FILE_NAME"],
            sim_dm=config["SIM_DM"],
            cnp2cnp_folder=config["cnp2cnp_FOLDER"],
            cnp2cnp_file=config["cnp2cnp_FILE"],
            true_tree_root_id=config["TRUE_TREE_ROOT_ID"],
            run_single_test=deepcopy(config["RUN_SINGLE_TEST"]),
        )

    def as_legacy_dict(self):
        return {
            "IN_FILE_NAME": self.in_file_name,
            "OUT_FILE_NAME": self.out_file_name,
            "SIM_DM": self.sim_dm,
            "cnp2cnp_FOLDER": self.cnp2cnp_folder,
            "cnp2cnp_FILE": self.cnp2cnp_file,
            "TRUE_TREE_ROOT_ID": self.true_tree_root_id,
            "RUN_SINGLE_TEST": deepcopy(self.run_single_test),
        }


def load_ctbs_config(config_path=CTBS_CONFIG_PATH):
    with open(config_path, "r") as f:
        loaded_config = json.load(f)

    config = deepcopy(DEFAULT_CTBS_CONFIG)
    config.update(loaded_config)
    return config


def load_ctbs_runtime_config(config_path=CTBS_CONFIG_PATH):
    return CtbsRuntimeConfig.from_mapping(load_ctbs_config(config_path))


def default_ctbs_runtime_config():
    return CtbsRuntimeConfig.from_mapping(DEFAULT_CTBS_CONFIG)


def _coerce_runtime_config(runtime_config=None):
    if runtime_config is None:
        return load_ctbs_runtime_config()
    if isinstance(runtime_config, CtbsRuntimeConfig):
        return runtime_config
    return CtbsRuntimeConfig.from_mapping(runtime_config)


def validate_distance_matrix(ids, matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Distance matrix must be two-dimensional.")
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError(f"Distance matrix must be square, got shape {matrix.shape}.")
    if ids is None:
        raise ValueError("Distance matrix ids are required for in-memory matrices.")
    if len(ids) != rows:
        raise ValueError(f"Distance matrix has {rows} rows but {len(ids)} ids.")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Distance matrix must be symmetric.")
    if not np.allclose(np.diag(matrix), np.zeros(rows)):
        raise ValueError("Distance matrix diagonal must be zero.")
    return list(ids), matrix


@dataclass(frozen=True)
class DistanceMatrix:
    ids: list | None = None
    matrix: object | None = None
    path: str | None = None

    def __post_init__(self):
        if self.path is None and self.matrix is None:
            raise ValueError("DistanceMatrix requires either an in-memory matrix or a path.")
        if self.matrix is not None:
            ids, matrix = validate_distance_matrix(self.ids, self.matrix)
            object.__setattr__(self, "ids", ids)
            object.__setattr__(self, "matrix", matrix)

    def build_tree_kwargs(self):
        if self.matrix is not None:
            return {
                "dist_matrix_path": None,
                "inids": self.ids,
                "indm": self.matrix,
            }
        return {"dist_matrix_path": self.path}


def unique_cells_by_cell_id(cells):
    unique = {}
    for cell in cells:
        if cell.cell_id not in unique:
            unique[cell.cell_id] = cell
    return list(unique.values())


def _trivial_distance_matrix(cells):
    ids = [cell.get_id() for cell in cells]
    return DistanceMatrix(ids=ids, matrix=np.zeros((len(ids), len(ids)), dtype=float))


class DistanceProvider:
    def compute(self, cells):
        raise NotImplementedError


@dataclass(frozen=True)
class SuppliedDistanceProvider(DistanceProvider):
    ids: list
    matrix: object

    def compute(self, cells):
        return DistanceMatrix(ids=self.ids, matrix=self.matrix)


@dataclass(frozen=True)
class Cnp2CnpPairwiseDistanceProvider(DistanceProvider):
    runtime_config: CtbsRuntimeConfig
    max_threads: int | None = None

    def compute(self, cells):
        if len(cells) <= 1:
            return _trivial_distance_matrix(cells)
        ids, matrix = distance_matrix_from_biopsy(
            cells,
            max_threads=self.max_threads,
            runtime_config=self.runtime_config,
        )
        return DistanceMatrix(ids=ids, matrix=matrix)


@dataclass(frozen=True)
class Cnp2CnpFileDistanceProvider(DistanceProvider):
    runtime_config: CtbsRuntimeConfig

    def compute(self, cells):
        if len(cells) <= 1:
            return _trivial_distance_matrix(cells)
        to_file(self.runtime_config.in_file_name, cells)
        use_cnp2cnp_to_compute_dist_matrix(
            self.runtime_config.in_file_name,
            runtime_config=self.runtime_config,
        )
        return DistanceMatrix(path=self.runtime_config.out_file_name)


def default_distance_provider(parallel=False, runtime_config=None, max_threads=None):
    runtime_config = _coerce_runtime_config(runtime_config)
    if parallel:
        return Cnp2CnpPairwiseDistanceProvider(runtime_config, max_threads=max_threads)
    return Cnp2CnpFileDistanceProvider(runtime_config)


class Timer:
    def __init__(self, label, collector=None, verbose=False):
        self.label = label
        self.collector = collector  # collector is a dict for the current run
        self.verbose = verbose

    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        duration = time.perf_counter_ns() - self.start
        if self.collector is not None:
            self.collector[self.label] = duration
        if self.verbose:
            print(f"{self.label}: {duration/1e6:.3f} ms")

#  print to file in a format that is compatible with cnp2cnp tool
def to_file(file, cells):
    with open(file, 'w') as f:
        for c in cells:
            f.write(">" + str(c.get_id()) + "\n")
            f.write(c.get_cnp() + "\n")


def _compute_pair(args):
    c, d, i, j, runfile = args
    input_str = f">{c.get_id()}\n{c.get_cnp()}\n>{d.get_id()}\n{d.get_cnp()}\n"
    dist = use_cnp2cnp_to_compute_pairwise_distance(input_str, runfile=runfile)
    return i, j, dist


def distance_matrix_from_biopsy(cells, max_threads=None, runtime_config=None):
    """
    Build a distance matrix for a list of cells using cnp2cnp.
    """
    n = len(cells)
    ids = [c.get_id() for c in cells]
    dist_matrix = np.zeros((n, n), dtype=float)
    if n <= 1:
        return ids, dist_matrix
    runtime_config = _coerce_runtime_config(runtime_config)

    pairs = [
        (cells[i], cells[j], i, j, runtime_config.cnp2cnp_file)
        for i in range(n)
        for j in range(i + 1, n)
    ]

    with ProcessPoolExecutor(max_workers=max_threads) as executor:
        for i, j, dist in executor.map(_compute_pair, pairs):
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return ids, dist_matrix


def use_cnp2cnp_to_compute_pairwise_distance(str_in, runfile=None, runtime_config=None):
    if runfile is None:
        runfile = _coerce_runtime_config(runtime_config).cnp2cnp_file
    pypath = str(sys.executable)

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(str_in)
        tmp.flush()
        infile_path = tmp.name

    out = subprocess.run(
        [pypath, runfile, "-m", "dist", "-i", infile_path],
        capture_output=True, text=True, check=True
    )
    return out.stdout


def use_cnp2cnp_to_compute_dist_matrix(sample=None, folder=None, runfile=None,
                                       output=None, runtime_config=None):
    """
    Execute cnp2cnp. Input CNPs of cells and obtain evolutionary distance matrix of given cells.

    Parameters
    ----------
    sample : string
        name of the file that contains information about the cell sample; every cell is described in two lines:
        first describing id of the cell, begins with ">" (in example ">cell_0")
        second contains CNP profile as a list of values in a form "value1,value2,...,valueN",
        where N is the length of CNP
    folder : string
        name of the folder that contains cnp2cnp tool described in
        https://doi.org/10.1186/s12864-020-6611-3
        to manually set path where input file will be copied; output file generated before copied back;
    runfile : string
        name of the file to execute (cnp2cnp.py)
        to manually set path to the file
    output : string
        name of the output file that will be generated in location set in argument 'folder',
        and copied to current location.
        first line of the file contains the number N of cells
        the following N lines represent distance matrix, each line consist of the id of the cell
        and N values that are evolutionary distances to the corresponding cells;
        evolutionary distance is the minimal number of events to transform on cell (CNP) to another cell (CNP)

    Returns
    -------
        In the current directory generates a file which name is set in argument 'output'.
        The file will contain distance matrix of cells, which CNPs are given in file described by argument 'input'.
        The file is generated by an external tool cnp2cnp.
    """
    runtime_config = _coerce_runtime_config(runtime_config)
    sample = runtime_config.in_file_name if sample is None else sample
    folder = runtime_config.cnp2cnp_folder if folder is None else folder
    runfile = runtime_config.cnp2cnp_file if runfile is None else runfile
    output = runtime_config.out_file_name if output is None else output

    shutil.copy(sample, folder)     # copy file to the cnp2cnp project
    cnp2cnp_in = os.path.join(folder, sample)
    cnp2cnp_out = os.path.join(folder, output)
    pypath = str(sys.executable)
    # compute distance matrix for cnps using cnp2cnp
    subprocess.run([pypath, runfile, "-m", "matrix", "-i", cnp2cnp_in, "-o", cnp2cnp_out])
    # sample use: python cnp2cnp.py -m matrix -i examples/probka1.txt -o examples/o.txt
    shutil.copy(cnp2cnp_out, os.getcwd())   # copy file with results back


def get_cell_manualy(cell_list, value):
    for cell in cell_list:
        if cell.get_id() == value:
            return cell
    return None

def show_cells(cell_list):
    for cell_l in cell_list:
        print("Biopsy: ", [cell.cell_id for cell in cell_l])

def _actual_root(tree):
    roots = [node for node, indegree in tree.in_degree() if indegree == 0]
    if len(roots) != 1:
        raise ValueError(f"Tree must have exactly one root, found {len(roots)}")
    return roots[0]

def _run_simulation(config, bedfile, seed, simulator_with_loaded_tree, time_collector):
    if simulator_with_loaded_tree:
        return simulator_with_loaded_tree

    if bedfile is not None:
        sim = CancerCellEvolutionSimulator(config, bedfile, seed=seed)
    else:
        sim = CancerCellEvolutionSimulator(config, seed=seed)

    if time_collector is not None:
        with Timer("Core simulation: ", time_collector):
            sim.run_simulation()
    else:
        sim.run_simulation()

    print("Simulation finished. Generated cell evolution tree total nodes:", len(sim.tree.nodes()))
    return sim

def _perform_biopsies(sim, biopsy_generations, biopsy_size, biopsy_size_scalable, seed,
                      compare_dm, runtime_config):
    cell_lists, all_in_one_sample = [], [[]]

    for b_gen in biopsy_generations:
        biopsy = sim.perform_biopsy(
            biopsy_size=biopsy_size,
            biopsy_size_scalable=biopsy_size_scalable,
            generation=b_gen,
            seed=seed
        )
        if biopsy: # we assume biopsy has at least one cell
            cell_lists.append(biopsy)
            all_in_one_sample[0] += biopsy
        else:
            print(f"Biopsy sample from generation {b_gen} has no cells. Skipping.")

    if compare_dm:
        sim.to_distance_matrix(runtime_config.sim_dm, [x.cell_id for x in all_in_one_sample[0]])

    print("Number of biopsy cells:", len(all_in_one_sample[0]))
    return cell_lists, all_in_one_sample

def _handle_small_biopsy(time_collector, min_total_cells=MIN_TOTAL_BIOPSY_CELLS):
    print(f"Total number of cells in biopsy less than {min_total_cells}.")
    if time_collector is not None:
        for key in ["Computing cnp2cnp distance matrix: ", "Clear CNPs: ", "GRF our: ", "GRF NJ: "]:
            time_collector[key] = 0

def _compute_distance_matrix(all_in_one_sample, parallel, time_collector, runtime_config, distance_provider=None):
    # for parallel case single distances are being computed
    # for not parallel we write biopsy to cnp2cnp format file, and proces that
    unique_cells = unique_cells_by_cell_id(all_in_one_sample[0])
    if distance_provider is None:
        distance_provider = default_distance_provider(parallel=parallel, runtime_config=runtime_config)

    if time_collector is not None:
        with Timer("Computing cnp2cnp distance matrix: ", time_collector):
            distance_matrix = distance_provider.compute(unique_cells)
    else:
        distance_matrix = distance_provider.compute(unique_cells)
    return distance_matrix

def _reconstruct_and_evaluate(sim, seed, cell_lists, all_in_one_sample, r_dist, visualize,
                              clear_cnps, parallel, write_newick, reconstruction_algorithm,
                              biopsy_guided_config, inid, indm, time_collector,
                              runtime_config=None, distance_matrix=None):
    runtime_config = _coerce_runtime_config(runtime_config)
    if distance_matrix is None:
        if parallel:
            distance_matrix = DistanceMatrix(ids=inid, matrix=indm)
        else:
            distance_matrix = DistanceMatrix(path=runtime_config.out_file_name)
    cl, osl = deepcopy(cell_lists), deepcopy(all_in_one_sample)
    show_cells(cell_lists)

    # Optional visualization
    if visualize:
        sim.plot_tree(biopsy_lists=cell_lists, highlight_nodes=all_in_one_sample[0],
                      legend_y_offset=-170, output_file="simulated_tree")

        # true tree
        only_nodes = [c.cell_id for c in all_in_one_sample[0]]
        sim.plot_tree(biopsy_lists=cell_lists, legend_y_offset=-170,
                      highlight_nodes=all_in_one_sample[0],extended=False,
                      only_nodes=only_nodes,node_numbers=True,output_file="true_tree")

    # # Options for True tree pic
    # only_nodes = [0, 1, 3, 5, 4, 7, 13, 12, 19]
    # if visualize:
    #     sim.plot_tree(biopsy_lists=cell_lists, legend_y_offset=-170,
    #                   highlight_nodes=all_in_one_sample[0], output_file="simulated_tree")
    #     sim.plot_tree(biopsy_lists=cell_lists,legend_y_offset=-170,
    #                   highlight_nodes=all_in_one_sample[0],extended=False,
    #                   only_nodes=only_nodes,node_numbers=True,output_file="true_tree")
    # # if visualize:
    # #     sim.plot_tree(biopsy_lists=cell_lists,legend_y_offset=-170,
    # #                   highlight_nodes=all_in_one_sample[0])

    # # Clear CNPs if requested
    # if clear_cnps:
    #     with Timer("Clear CNPs: ", time_collector) if time_collector else contextlib.nullcontext():
    #         true_tree_simplified = sim.tree_without_CNPs()
    #         for lst in (cl + [osl[0]]):
    #             for cell in lst:
    #                 cell.genome = np.array([], dtype=int)
    # else:
    #     true_tree_simplified = sim.tree

    if clear_cnps:    # clear CNPs
        if time_collector is not None:
            with Timer("Clear CNPs: ", time_collector):
                true_tree_simplified = sim.tree_without_CNPs()  # clears simulated tree
                for cell_list in cl:                            # clears biopsy
                    for cell in cell_list:
                        cell.genome = np.array([], dtype=int)
                for cell in osl[0]:
                    cell.genome = np.array([], dtype=int)
    else:
        true_tree_simplified = sim.tree

    # --- unified build config ---
    build_kwargs = {"r": r_dist}
    build_kwargs.update({"seed": seed})
    build_kwargs.update(distance_matrix.build_tree_kwargs())
    if reconstruction_algorithm:
        build_kwargs["neighbor_joining"] = reconstruction_algorithm

    # --- build trees ---
    njtree, nj_info, _returned_root_nj = build_evolution_tree(osl, only_nj=True, **build_kwargs)
    rec_build_kwargs = build_kwargs.copy()
    if biopsy_guided_config is not None:
        rec_build_kwargs["biopsy_guided_config"] = biopsy_guided_config
    tree, rt_info, _returned_root_rt = build_evolution_tree(cl, **rec_build_kwargs)
    actual_root_nj = _actual_root(njtree)
    actual_root_rt = _actual_root(tree)

    if write_newick:
        print("Newick simulated:", to_newick(sim.tree))
        print("Reconstructed:", to_newick(tree))
        print("NJ tree:", to_newick(njtree))

    # Visualization (optional)
    if visualize:
        visualize_tree_plotly(tree, rt_info, output_file="reconstructed.html")
        visualize_tree_plotly(njtree, nj_info, output_file="nj.html")

    # if visualize:
    #     lno = {2:[8,6], 1:[20,21,22,25,16,30], 0:[50,32,54,34,56,57,21,38,65,43,71,48]}
    #     visualize_tree_plotly(tree, rt_node_info_for_plots, level_node_ordering=lno, output_file="reconstructed.html")
    #     lno1 = {0:[50,32,54,34,20,56,57,21,22,38,8,65,25,43,16,6,71,30,48]}
    #     visualize_tree_plotly(njtree, nj_node_info_for_plots, level_node_ordering=lno1, output_file="nj.html")


    # --- Evaluate GRF distances ---
    if time_collector is not None:
        with Timer("GRF our: ", time_collector):
            ret1 = grf_tree(true_tree_simplified, runtime_config.true_tree_root_id, tree, actual_root_rt)
        with Timer("GRF NJ: ", time_collector):
            ret2 = grf_tree(true_tree_simplified, runtime_config.true_tree_root_id, njtree, actual_root_nj)
    else:
        ret1 = grf_tree(true_tree_simplified, runtime_config.true_tree_root_id, tree, actual_root_rt)
        ret2 = grf_tree(true_tree_simplified, runtime_config.true_tree_root_id, njtree, actual_root_nj)

    print("GRF - reconstructed:", ret1)
    print("GRF - NJ:", ret2)

    return true_tree_simplified, tree, njtree


def run_single_test(config="config_telomeric.json", bedfile="bed like config sample.csv", seed=777,
                    biopsy_size=2, biopsy_size_scalable=None, biopsy_generations=[5,7,9], r_dist=4,
                    visualize=False, time_collector=None, clear_cnps=False, compare_dm=False,
                    write_newick=False, simulator_with_loaded_tree=None, parallel=False,
                    reconstruction_algorithm=None, biopsy_guided_strategy=None,
                    biopsy_guided_config=None, runtime_config=None, distance_provider=None):
    """
    Runs one test that consists of simulation, biopsy, tree reconstruction and tree evaluation.

    Parameters
    ----------
    config  - input configuration file for global parameters for the simulator
    bedfile - optional additional configuration file for positional parameters for the simulator
    seed    - seed for random number generator
    biopsy_size             - size of biopsy (number of cells sampled from given level)
    biopsy_size_scalable    - size of biopsy (percentage of cells sampled from given level)
    biopsy_generatons       - list of levels of the tree (generations numbers) used for biopsy
    r_dist  - the proximity radius for tree reconstruction
    visualize       - whether to visualize the simulation results
    time_collector  - whether to time the simulation runs
    clear_cnps  - whether to clear cnps (potential optimization - makes the simulation tree light)
    compare_dm      - whether to output distance matrix of simulated tree cells
    write_newick       - whether to output simulated tree and reconstructed tree in newick format
    simlulator_with_loaded_tree - for testing and repeatability, simlulator with loaded tree

    Returns
    -------
    The similarity values between simulated tree and reconstructed tree,
    and between simulated tree and NJ-reconstructed tree.

    """
    runtime_config = _coerce_runtime_config(runtime_config)

    if biopsy_guided_config is None:
        biopsy_guided_config = resolve_biopsy_guided_config(biopsy_guided_strategy)

    # 1. Simulation phase
    sim = _run_simulation(config, bedfile, seed, simulator_with_loaded_tree, time_collector)

    # 2. Biopsy phase
    cell_lists, all_in_one_sample = _perform_biopsies(sim, biopsy_generations, biopsy_size,
                                                      biopsy_size_scalable, seed, compare_dm,
                                                      runtime_config)

    if len(all_in_one_sample[0]) < MIN_TOTAL_BIOPSY_CELLS:
        _handle_small_biopsy(time_collector)
        return

    # 3. Distance matrix computation
    distance_matrix = _compute_distance_matrix(
        all_in_one_sample,
        parallel,
        time_collector,
        runtime_config,
        distance_provider=distance_provider,
    )

    # 4. Tree reconstruction and evaluation
    return _reconstruct_and_evaluate(
        sim,
        seed,
        cell_lists,
        all_in_one_sample,
        r_dist,
        visualize,
        clear_cnps,
        parallel,
        write_newick,
        reconstruction_algorithm,
        biopsy_guided_config,
        distance_matrix.ids,
        distance_matrix.matrix,
        time_collector,
        runtime_config,
        distance_matrix,
    )


def run_single_test_timed(seed, both=True, **kwargs):
    """
    Wrapper for run_single_test to measure the time for the cases with clear_cnps optimization on and off.

    Parameters
    ----------
    seed        passed to run_single_test
    both        if true executes run_single_test two times with clear_cnps optimization on and off;
                if false executes run_single_test without clear_cnps optimization;
    kwargs      parameters passed to run_single_test

    Returns
    -------
    Dictionaries with times of executions of parts of test (computing distance matrix, GRF distances ...)
    """
    run_timings_no_opt = {}
    with Timer("Total", run_timings_no_opt):
        run_single_test(seed=seed, time_collector=run_timings_no_opt, clear_cnps=False, **kwargs)

    run_timings_with_opt = {}
    if both:
        with Timer("Total", run_timings_with_opt):
            run_single_test(seed=seed, time_collector=run_timings_with_opt, clear_cnps=True, **kwargs)

    return run_timings_no_opt, run_timings_with_opt


def check_clearcnp_optimizaton(how_many=100, both=True, seeds=None, **kwargs):
    """
    Runner

    Parameters
    ----------
    how_many    number of tests to run
    both        if true runs pair of test with clear_cnps optimization on and off;
                otherwise runs one test without optimization
    seeds       seeds for executions of run_single_test; if not given, randomly selected here
    kwargs      parameters passed to run_single_test

    Returns
    -------
    Prints summary of the tests.

    """
    if not seeds:
        seeds = [random.randint(0, 1000) for _ in range(how_many)]
        seeds = [696]

    all_runs_no_opt = []
    all_runs_with_opt = []

    for s in seeds:
        print(f"\nTesting seed: {s}")
        run_no_opt, run_with_opt = run_single_test_timed(seed=s, biopsy_size_scalable=0.5, both=both,
                                                         biopsy_generatons=[4, 6, 8], r_dist=4, **kwargs)
        all_runs_no_opt.append(run_no_opt)
        if both:
            all_runs_with_opt.append(run_with_opt)

    def average_timings(runs):
        all_keys = {k for run in runs for k in run.keys()}
        avg_dict = {}
        for key in sorted(all_keys):
            avg_dict[key] = sum(run.get(key, 0) for run in runs) / len(runs) / 1e6
        without_cnp_avg = sum(
            run["Total"] - run.get("Computing cnp2cnp distance matrix: ", 0) for run in runs
        ) / len(runs) / 1e6
        return avg_dict, without_cnp_avg

    avg_no_opt_dict, avg_no_opt_total = average_timings(all_runs_no_opt)
    if both:
        avg_with_opt_dict, avg_with_opt_total = average_timings(all_runs_with_opt)

    print("\n--- Average durations WITHOUT optimization (ms) ---")
    for k, v in avg_no_opt_dict.items():
        print(f"{k:<35}: {v:.3f}")
    print(f"Total without cnp call{' ' * 10}: {avg_no_opt_total:.3f}")

    if both:
        print("\n--- Average durations WITH optimization (ms) ---")
        for k, v in avg_with_opt_dict.items():
            print(f"{k:<35}: {v:.3f}")
        print(f"Total without cnp call{' ' * 10}: {avg_with_opt_total:.3f}")


if __name__ == "__main__":
    # seeds = [56, 777, 727, 7, 77, 22, 32]
    # for s in seeds:
    #     run_single_test(config="config_telomeric.json", bedfile=None, seed=s,
    #                     biopsy_size=0.5, biopsy_generatons=[5, 7, 9], r_dist=4, visualize=False)

    # r_list = [20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    # for r in r_list:
    #     print("Running simulation for r=", r)
    #     run_single_test(config="config_telomeric.json", bedfile=None, seed=22,
    #                 biopsy_size=0.5, biopsy_generatons=[5, 7, 9], r_dist=r, visualize=False)

    # timing_data = defaultdict(list)
    # seeds = [56, 777, 7, 77, 22, 32, 727, 0, 100, 1000]
    # for s in seeds:
    #     with Timer("Total", timing_data):
    #         run_single_test(config="config_telomeric.json", bedfile=None, seed=s, biopsy_size=2,
    #                         biopsy_generatons=[4, 7], r_dist=4, visualize=False,
    #                         time_collector=timing_data, clear_cnps=True)
    #
    # print("\nAverage durations (ms):")
    # for key, times in timing_data.items():
    #     avg_ms = sum(times) / len(times) / 1e6
    #     print(f"{key:<15}: {avg_ms:.3f} ms")
    # run_single_test(config="config_for_pic.json", bedfile="pic.csv", seed=727,
    #                 biopsy_size_scalable=0.5, biopsy_generatons=[4, 7], r_dist=4,
    #                 visualize=True)

    # run_single_test(config="config_for_pic.json", bedfile="pic.csv", seed=727,
    #                 biopsy_size_scalable=0.5, biopsy_generatons=[4, 6, 8], r_dist=4,
    #                 visualize=True)

    # run_single_test(seed=727, config="test/data/config_for_pic.json", bedfile="test/data/pic.csv",
    #                       biopsy_size_scalable=0.5, biopsy_generatons=[4, 6, 8], r_dist=4, visualize=True)

    # check_clearcnp_optimizaton(how_many=1, seeds=[773], config="test/data/config_for_pic.json",
    #                            bedfile="test/data/pic.csv", parallel=True, both=False)

    # run_single_test(seed=773, config="test/data/config_for_pic.json",
    #                            bedfile="test/data/pic.csv") #, parallel=True)

    # run_single_test(seed=773, config="test/data/config_for_pic.json", bedfile="test/data/pic.csv",
    #                 biopsy_size_scalable=0.5, biopsy_generations=[4, 6, 8], r_dist=4, write_newick=True,
    #                 reconstruction_algorithm=neighbor_joining_full)

    # seed 35 !!!
    # seed 632
    runtime_config = load_ctbs_runtime_config()
    run_config = runtime_config.run_single_test.copy()
    run_config["reconstruction_algorithm"] = resolve_reconstruction_algorithm(
        run_config.get("reconstruction_algorithm")
    )
    run_config["biopsy_guided_config"] = resolve_biopsy_guided_config(
        run_config.pop("biopsy_guided_strategy", None)
    )
    a, b, c = run_single_test(**run_config, runtime_config=runtime_config)

    biopsy_nodes_ids = get_biopsy_nodes_ids(b, c)

    out = evaluate_4(a, b, restrict_labels=biopsy_nodes_ids, print_debug=True)
    print(out)

    out = evaluate_4(a, c, restrict_labels=biopsy_nodes_ids, print_debug=True)
    print(out)
    # print(c.edges)
    # print(len(c.nodes))
    # l = [(named_label(c, x), named_label(c, y)) for x, y in c.edges]
    # print(l)
    # roots = [n for n, indeg in c.in_degree() if indeg == 0]
    # if len(roots) != 1:
    #     raise ValueError(f"Tree must have exactly one root (found {len(roots)})")
    # root = roots[0]
    # vizualize_nx_tree(c)

    # run_single_test(seed=773, config="test/data/config_for_pic.json", bedfile="test/data/pic.csv",
    #                 biopsy_size_scalable=0.5, biopsy_generatons=[3, 5, 7, 9], r_dist=4)

    # check_clearcnp_optimizaton(how_many=10, config="test/data/config100.json", bedfile=None)
    # check_clearcnp_optimizaton(how_many=1, config="test/data/config_for_pic.json",
    #                            bedfile=None, both=False, parallel=True)
