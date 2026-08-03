import copy
import itertools
from collections import Counter, defaultdict
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import csv

from ctbf_constraints import MIN_BIOPSY_CELLS_FROM_BIOPSY
from distance_semantics import validate_distance_matrix
from simulator_config import (
    DiscreteDistribution,
    load_simulator_inputs,
)
from simulator_events import (
    apply_event_sequence,
    count_edge_events,
    event_records_to_dicts,
    event_records_to_text,
    propose_event_sequence,
)

"""
Represents a unique type of cell (genotype) with:
  - genome: CN profile (e.g., numpy array)
  - node_id: unique identifier (for the tree)
  - cell_id: unique identifier for each cell (genotype)
"""


class Genotype:
    def __init__(self, genome, node_id, generation=None, cell_id=None):
        self.genome = np.array(genome)
        self.node_id = node_id
        self.generation = generation
        self.cell_id = cell_id if cell_id is not None else node_id  # Default to node_id if no cell_id is provided

    def __repr__(self):
        return f"Genotype(ID={self.node_id}, Gen={self.generation}, cell_id={self.cell_id}, genome={self.genome})"

    def get_id(self):
        return self.cell_id

    # Outputs a string to match input cnp2cnp format
    def get_cnp(self):
        return ",".join(str(cnp) for cnp in self.genome)


class SimulationDiagnostics:
    """Owned counters for simulator attempts, viability, and state collisions."""

    def __init__(self):
        self.totals = Counter()
        self.by_generation = defaultdict(Counter)

    def increment(self, generation, name, amount=1):
        self.totals[name] += int(amount)
        self.by_generation[int(generation)][name] += int(amount)

    def snapshot(self):
        return {
            "totals": dict(sorted(self.totals.items())),
            "by_generation": {
                str(generation): dict(sorted(values.items()))
                for generation, values in sorted(self.by_generation.items())
            },
        }


"""
CTBF v2 CNP-state evolution simulator.

Strict JSON/BED parsing and immutable resolved inputs live in
``simulator_config``. Pure typed CNA proposal/application mechanics live in
``simulator_events``. This class owns population/tree state, identity, random
streams, biopsy sampling, diagnostics, provenance, and exports.
"""


class CancerCellEvolutionSimulator:
    _RNG_STREAM_NAMES = (
        "offspring",
        "event_initiation",
        "event_class",
        "footprint",
        "event_type",
        "gain_operator",
        "magnitude_additive",
        "magnitude_factor",
        "event_order",
        "wgd",
        "biopsy",
    )

    def __init__(self, config_source, genome_bed=None, seed=None):
        inputs = load_simulator_inputs(config_source, genome_bed)
        self.config = inputs.config
        self.genome_bins = inputs.genome_bins
        self.semantic_version = self.config.semantic_version
        self.config_sha256 = inputs.config_sha256
        self.bed_sha256 = inputs.bed_sha256
        self.config_source = inputs.config_source
        self.bed_source = inputs.bed_source

        if isinstance(seed, bool) or (seed is not None and not isinstance(seed, (int, np.integer))):
            raise ValueError("Simulator seed must be a non-negative integer or None.")
        if seed is not None and int(seed) < 0:
            raise ValueError("Simulator seed must be a non-negative integer or None.")
        seed_sequence = np.random.SeedSequence(None if seed is None else int(seed))
        self.seed = int(seed_sequence.entropy)
        child_sequences = seed_sequence.spawn(len(self._RNG_STREAM_NAMES))
        self._rng_streams = {
            name: np.random.default_rng(child_sequence)
            for name, child_sequence in zip(self._RNG_STREAM_NAMES, child_sequences)
        }
        self._rng_spawn_keys = {
            name: list(child_sequence.spawn_key)
            for name, child_sequence in zip(self._RNG_STREAM_NAMES, child_sequences)
        }

        self.genome_length = len(self.genome_bins)
        self.founder_genome = np.asarray(
            [genome_bin.initial_copy_number for genome_bin in self.genome_bins],
            dtype=int,
        )
        self.chromosome = tuple(genome_bin.chromosome for genome_bin in self.genome_bins)
        self.crucial_for_survival = tuple(genome_bin.crucial for genome_bin in self.genome_bins)
        self.num_generations = self.config.number_of_generations
        self.offspring_model = self.config.offspring_model
        self.offspring_param = self.config.offspring_parameter
        self.baseline_descendant_attempts = self.config.baseline_descendant_attempts
        self.representation_type = self.config.representation_type
        self.model_telomeric_regions = self.config.telomeric_instability_enabled
        self.model_crucial_for_survival = self.config.crucial_survival_enabled
        self.diagnostics = SimulationDiagnostics()

        self.tree = nx.DiGraph()
        founder = Genotype(genome=self.founder_genome, node_id=0, generation=0, cell_id=0)
        self.genotypes = {0: founder}
        self.tree.add_node(
            0,
            genome=self.founder_genome.tolist(),
            generation=0,
            cell_id=0,
        )
        self.tree.graph["simulator_provenance"] = self.provenance()
        self.node_counter = 1
        self._canonical_cell_id_by_genome = None

    def provenance(self):
        first_generator = next(iter(self._rng_streams.values()))
        return {
            "simulator_semantic_version": self.semantic_version,
            "seed": self.seed,
            "numpy_version": np.__version__,
            "bit_generator": type(first_generator.bit_generator).__name__,
            "rng_stream_spawn_keys": dict(self._rng_spawn_keys),
            "config_sha256": self.config_sha256,
            "bed_sha256": self.bed_sha256,
            "config_source": self.config_source,
            "bed_source": self.bed_source,
        }

    def diagnostics_snapshot(self):
        return self.diagnostics.snapshot()

    @classmethod
    def from_tree(cls, input_tree: nx.DiGraph):
        """
        Alternative constructor: build simulator directly from a given tree.

        Parameters
        ----------
        input_tree : nx.DiGraph
            A directed graph where each node has attributes:
            'genome', 'generation', 'cell_id', 'id'.

        Returns
        -------
        CancerCellEvolutionSimulator
        """
        self = cls.__new__(cls)
        # NetworkX ``copy`` is shallow for nested node/edge attributes. The
        # simulator owns its tree, including mutable genome and event records.
        self.tree = copy.deepcopy(input_tree)
        self.genotypes = {}
        self._canonical_cell_id_by_genome = None
        stored_provenance = self.tree.graph.get("simulator_provenance", {})
        self.semantic_version = stored_provenance.get(
            "simulator_semantic_version", "external-tree"
        )
        self.seed = None
        self._rng_streams = {"biopsy": np.random.default_rng()}
        self._rng_spawn_keys = {}
        self.config_sha256 = stored_provenance.get("config_sha256")
        self.bed_sha256 = stored_provenance.get("bed_sha256")
        self.config_source = stored_provenance.get("config_source")
        self.bed_source = stored_provenance.get("bed_source")
        self.diagnostics = SimulationDiagnostics()

        # Fill genotypes dict from nodes
        for node_id, data in self.tree.nodes(data=True):
            genome = np.array(data["genome"])  # genome as numpy array
            generation = data.get("generation")
            cell_id = data.get("cell_id", node_id)

            self.genotypes[node_id] = Genotype(
                genome=genome,
                node_id=node_id,
                generation=generation,
                cell_id=cell_id,
            )

        self.node_counter = max(self.genotypes.keys(), default=-1) + 1

        return self

    @staticmethod
    def _genome_key(genome):
        return tuple(int(value) for value in np.asarray(genome).tolist())

    def canonical_cell_id_by_genome(self):
        """
        Return the canonical cell_id for each observed genome.

        Copy-number events can return a lineage to an already observed genome.
        In that case the raw simulation tree can contain multiple cell_id values
        for the same genome. Biopsy/export consumers use the smallest observed
        cell_id as the canonical biological genotype id for that genome.
        """
        if self._canonical_cell_id_by_genome is None:
            canonical = {}
            by_cell_id = {}
            for genotype in self.genotypes.values():
                genome_key = self._genome_key(genotype.genome)
                cell_id = int(genotype.cell_id)
                if cell_id in by_cell_id and by_cell_id[cell_id] != genome_key:
                    raise ValueError(
                        f"cell_id {cell_id!r} maps to multiple genomes; "
                        "this is a hard simulator invariant error"
                    )
                by_cell_id[cell_id] = genome_key
                canonical[genome_key] = min(canonical.get(genome_key, cell_id), cell_id)
            self._canonical_cell_id_by_genome = canonical
        return self._canonical_cell_id_by_genome

    def canonical_cell_id_for_genome(self, genome):
        return self.canonical_cell_id_by_genome()[self._genome_key(genome)]

    def canonicalize_genotype_cell_id(self, genotype):
        canonical_cell_id = self.canonical_cell_id_for_genome(genotype.genome)
        if canonical_cell_id == genotype.cell_id:
            return genotype
        return Genotype(
            genome=genotype.genome.copy(),
            node_id=genotype.node_id,
            generation=genotype.generation,
            cell_id=canonical_cell_id,
        )

    def canonicalize_biopsy_genotypes(self, genotypes):
        canonicalized = []
        seen_cell_ids = set()
        for genotype in genotypes:
            canonical_genotype = self.canonicalize_genotype_cell_id(genotype)
            if canonical_genotype.cell_id in seen_cell_ids:
                continue
            seen_cell_ids.add(canonical_genotype.cell_id)
            canonicalized.append(canonical_genotype)
        return canonicalized

    def canonicalized_tree_by_genome(self):
        tree_copy = self.tree.copy()
        canonical = self.canonical_cell_id_by_genome()
        for _, data in tree_copy.nodes(data=True):
            if "genome" not in data:
                continue
            data["cell_id"] = canonical[self._genome_key(data["genome"])]
        return tree_copy

    def get_parameters_csv(self, file):
        """Export the resolved immutable v2 locus parameters as a wide CSV."""
        with open(file, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "ChromosomeNumber",
                    "Start",
                    "End",
                    "CN",
                    "CNA_EVENT_PROBABILITY",
                    "GAIN_GIVEN_CNA_PROBABILITY",
                    "INTERVAL_CNA_PROBABILITY",
                    "INTERVAL_GAIN_OPERATOR_PROBABILITIES",
                    "ADDITIVE_GAIN_LAMBDA",
                    "MULTIPLICATIVE_FACTOR_PROBABILITIES",
                    "TELOMERIC_INSTABILITY",
                    "CRUCIAL",
                ]
            )
            for genome_bin in self.genome_bins:
                writer.writerow(
                    [
                        genome_bin.chromosome,
                        genome_bin.start,
                        genome_bin.end,
                        genome_bin.initial_copy_number,
                        genome_bin.cna_event_probability,
                        genome_bin.gain_given_cna_probability,
                        genome_bin.interval_cna_probability,
                        self._distribution_text(genome_bin.interval_gain_operators),
                        genome_bin.additive_gain_lambda,
                        self._distribution_text(genome_bin.multiplicative_factors),
                        genome_bin.telomeric_instability,
                        int(genome_bin.crucial),
                    ]
                )

    @staticmethod
    def _distribution_text(distribution: DiscreteDistribution):
        return ",".join(
            f"{value}:{probability:g}"
            for value, probability in zip(
                distribution.values,
                distribution.probabilities,
            )
        )

    def create_bed_csv(self, file):
        """Export a strict BED-like v2 input with all resolved row parameters."""
        with open(file, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ChromosomeNumber", "Start", "End", "Parameters"])
            for genome_bin in self.genome_bins:
                parameters = ";".join(
                    [
                        f"CN={genome_bin.initial_copy_number}",
                        f"CNA_EVENT_PROBABILITY={genome_bin.cna_event_probability:g}",
                        f"GAIN_GIVEN_CNA_PROBABILITY={genome_bin.gain_given_cna_probability:g}",
                        f"INTERVAL_CNA_PROBABILITY={genome_bin.interval_cna_probability:g}",
                        "INTERVAL_GAIN_OPERATOR_PROBABILITIES="
                        f"{self._distribution_text(genome_bin.interval_gain_operators)}",
                        f"ADDITIVE_GAIN_LAMBDA={genome_bin.additive_gain_lambda:g}",
                        "MULTIPLICATIVE_FACTOR_PROBABILITIES="
                        f"{self._distribution_text(genome_bin.multiplicative_factors)}",
                        f"CRUCIAL={int(genome_bin.crucial)}",
                    ]
                    + (
                        [f"TELOMERIC_INSTABILITY={genome_bin.telomeric_instability:g}"]
                        if self.model_telomeric_regions
                        else []
                    )
                )
                writer.writerow(
                    [
                        genome_bin.chromosome,
                        genome_bin.start,
                        genome_bin.end,
                        parameters,
                    ]
                )

    def _sample_additional_offspring_count(self):
        rng = self._rng_streams["offspring"]
        if self.offspring_model == "constant":
            return int(self.offspring_param)
        if self.offspring_model == "uniform_range":
            minimum, maximum = self.offspring_param
            return int(rng.integers(minimum, maximum + 1))
        if self.offspring_model == "poisson":
            return int(rng.poisson(float(self.offspring_param)))
        raise ValueError("Unsupported CTBF v2 OFFSPRING_MODEL.")

    def _apply_copy_number_events(self, genome):
        proposals = propose_event_sequence(
            self.genome_bins,
            wgd_probability=self.config.wgd_probability,
            rng_streams=self._rng_streams,
        )
        return apply_event_sequence(
            genome,
            proposals,
            self.genome_bins,
            crucial_survival_enabled=self.model_crucial_for_survival,
        )

    """
    Run the simulation for a specified number of generations.
    In each generation:
      1. Generate offspring for existing genotypes.
    """

    def run_simulation(self):
        for gen in range(1, self.num_generations + 1):
            self._spawn_children(current_generation=gen)
        self.tree.graph["simulation_diagnostics"] = self.diagnostics_snapshot()

    """
    For each genotype, draw the number of offspring based on Poisson distribution.
    For each child, create a new genome by copying the parent and applying mutations.
    """

    def _spawn_children(self, current_generation):
        new_genotypes = []
        parent_genotypes = sorted(
            (
                genotype
                for genotype in self.genotypes.values()
                if genotype.generation == current_generation - 1
            ),
            key=lambda genotype: genotype.node_id,
        )
        seen_genomes = {}
        for genotype in parent_genotypes:
            additional_children = self._sample_additional_offspring_count()
            number_of_attempts = self.baseline_descendant_attempts + additional_children
            for _ in range(number_of_attempts):
                self.diagnostics.increment(current_generation, "attempted_children")
                event_result = self._apply_copy_number_events(genotype.genome)
                attempted_records = event_result.attempted_records
                self.diagnostics.increment(
                    current_generation,
                    "attempted_event_records",
                    len(attempted_records),
                )
                self.diagnostics.increment(
                    current_generation,
                    "effective_event_applications",
                    sum(record.effective for record in attempted_records),
                )
                self.diagnostics.increment(
                    current_generation,
                    "ineffective_event_applications",
                    sum(not record.effective for record in attempted_records),
                )
                self.diagnostics.increment(
                    current_generation,
                    "sampled_wgd_events",
                    sum(
                        record.event_class == "whole_genome_doubling"
                        for record in attempted_records
                    ),
                )

                if event_result.rejected_by_crucial_survival:
                    self.diagnostics.increment(current_generation, "crucial_rejections")
                    continue

                child_genome = event_result.genome
                if child_genome is None:  # pragma: no cover - result invariant
                    raise RuntimeError("Non-rejected event application returned no genome.")
                self.diagnostics.increment(current_generation, "viable_children")
                if event_result.net_zero_sequence:
                    self.diagnostics.increment(current_generation, "net_zero_sequences")

                if self.representation_type == "representative":
                    genome_tuple = self._genome_key(child_genome)
                    if genome_tuple in seen_genomes:
                        self.diagnostics.increment(
                            current_generation,
                            "representative_collisions",
                        )
                        if seen_genomes[genome_tuple] != genotype.node_id:
                            self.diagnostics.increment(
                                current_generation,
                                "cross_parent_representative_collisions",
                            )
                        continue
                    seen_genomes[genome_tuple] = genotype.node_id

                edge_records = event_result.edge_records
                event_dicts = event_records_to_dicts(edge_records)
                event_summary = event_records_to_text(edge_records)
                child_cell_id = self.node_counter if edge_records else genotype.cell_id
                child = Genotype(
                    genome=child_genome,
                    node_id=self.node_counter,
                    generation=current_generation,
                    cell_id=child_cell_id,
                )
                new_genotypes.append((genotype.node_id, child))
                self.tree.add_node(
                    self.node_counter,
                    genome=child_genome.tolist(),
                    generation=current_generation,
                    cell_id=child_cell_id,
                )
                self.tree.add_edge(
                    genotype.node_id,
                    self.node_counter,
                    events=event_dicts,
                    events_text=event_summary,
                    event_count=len(event_dicts),
                )
                self.diagnostics.increment(current_generation, "retained_children")
                self.diagnostics.increment(
                    current_generation,
                    "edge_event_records",
                    len(event_dicts),
                )
                self.node_counter += 1

        for _, child in new_genotypes:
            self.genotypes[child.node_id] = child
        if new_genotypes:
            self._canonical_cell_id_by_genome = None

    def perform_biopsy(self, generation, biopsy_size=0, biopsy_size_scalable=None, seed=None):
        """
        Randomly selects unique genotypes from a given generation.

        When biopsy_size_scalable is provided, a non-empty generation always
        contributes at least one sampled cell, even if the scaled size floors to
        zero. Empty generations still return an empty list.

        Returns a list of selected genotype objects.
        """
        # Get genotypes from the specified generation
        genotypes_from_generation = [g for g in self.genotypes.values() if g.generation == generation]

        if biopsy_size_scalable is not None:
            available_count = len(genotypes_from_generation)
            biopsy_size = 0 if available_count == 0 else max(
                MIN_BIOPSY_CELLS_FROM_BIOPSY,
                int(biopsy_size_scalable * available_count),
            )
            biopsy_size = min(biopsy_size, available_count)
        else:
            # If there are fewer genotypes than the biopsy size, return all available.
            biopsy_size = min(biopsy_size, len(genotypes_from_generation))

        if seed is None:
            rng = self._rng_streams["biopsy"]
        else:
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
                raise ValueError("Biopsy seed must be a non-negative integer or None.")
            rng = np.random.default_rng(int(seed))
        sampled_genotypes = rng.choice(
            genotypes_from_generation,
            size=biopsy_size,
            replace=False,
        )

        return self.canonicalize_biopsy_genotypes(list(sampled_genotypes))

    def tree_without_CNPs(self):
        """Return a copy of the tree where each node's genome is replaced with None to free memory."""
        tree_copy = self.tree.copy()
        for node in tree_copy.nodes:
            tree_copy.nodes[node]['genome'] = None
        return tree_copy

    def to_distance_matrix(self, output_filename=None, node_list=None, labels=None):
        """
        Compute and save a distance matrix of tree nodes in PHYLIP format
        based on the number of evolutionary events along the path between nodes.
        Uses cell_id for labeling unless explicit biological labels are supplied.

        Parameters
        ----------
        output_filename : str, optional
            Path to output PHYLIP-style distance matrix file. If omitted, only
            the validated in-memory result is returned.
        node_list : list, optional
            If provided, only include these node IDs in the distance matrix.
        labels : list, optional
            Labels corresponding one-to-one with ``node_list``. This keeps
            simulator occurrence identity (node_id) separate from canonical
            biological CNP identity (cell_id).
        """
        undirected_tree = self.tree.to_undirected()

        # Filter nodes
        if node_list is not None:
            nodes = list(node_list)
            if len(nodes) != len(set(nodes)):
                raise ValueError("Simulator distance node_list contains duplicates.")
            missing_nodes = [node for node in nodes if node not in self.tree]
            if missing_nodes:
                raise ValueError(
                    "Simulator distance node_list contains unknown node ids: "
                    f"{missing_nodes!r}."
                )
        else:
            nodes = list(self.tree.nodes())

        n = len(nodes)

        # Get cell_ids for labeling
        if labels is None:
            cell_ids = [self.tree.nodes[node]["cell_id"] for node in nodes]
        else:
            cell_ids = list(labels)
            if len(cell_ids) != n:
                raise ValueError(
                    "Simulator distance labels must match node_list length."
                )

        # Precompute all pairwise distances
        dist_matrix = np.zeros((n, n), dtype=int)
        for i, j in itertools.combinations(range(n), 2):
            src, tgt = nodes[i], nodes[j]
            path = nx.shortest_path(undirected_tree, source=src, target=tgt)

            total_events = 0
            for u, v in zip(path[:-1], path[1:]):
                edge_data = self.tree.get_edge_data(u, v) or self.tree.get_edge_data(v, u)
                if edge_data is not None:
                    total_events += count_edge_events(edge_data.get("events"))

            dist_matrix[i, j] = total_events
            dist_matrix[j, i] = total_events

        cell_ids, dist_matrix = validate_distance_matrix(cell_ids, dist_matrix)

        if output_filename is not None:
            with open(output_filename, "w") as f:
                f.write(f"{n}\n")  # number of nodes first
                for i, cid in enumerate(cell_ids):
                    f.write(f"{str(cid):<10}")
                    f.write(" ".join(str(dist) for dist in dist_matrix[i]))
                    f.write("\n")
        return cell_ids, dist_matrix

    def plot_tree(self, title="Population Evolution", output_file=None,
                  biopsy_lists=None, highlight_nodes=None, node_numbers=False, x_scale=1.25, y_scale=1,
                  legend_x_offset=95, legend_y_offset=-200, extended=True, only_nodes=None):
        G = self.tree  # Assuming self.tree is a NetworkX graph

        # Compute positions using Graphviz layout
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

        # Scale positions to flatten and widen the tree
        scale_x = x_scale
        scale_y = y_scale
        scaled_pos = {node: (x * scale_x, y * scale_y) for node, (x, y) in pos.items()}

        # Extract node and edge data
        node_x, node_y = [], []
        node_labels = []
        hover_labels = []
        for node, data in G.nodes(data=True):
            x, y = scaled_pos[node]
            node_x.append(x)
            node_y.append(y)


            # Full hover label
            gen_num = data.get("generation", 0)
            gen_str = data.get("genome", [])
            cell_id = data.get("cell_id", "N/A")  # Default if cell_id is missing
            hover_text = (f"Cell ID={node}; "
                          f"Genotype ID={cell_id}; "  # Include cell_id in hover text
                          #f"Gen={gen_num}<br>"
                          f"{gen_str}")

            hover_labels.append(hover_text)

        # Prepare for biopsy highlighting
        biopsy_highlight_x = []
        biopsy_highlight_y = []
        biopsy_colors = []

        if biopsy_lists is not None:
            # Define a set of n shades of red for highlighting (lighter to darker red)
            red_shades = ['lightgoldenrodyellow', '#ffcccc', '#ebd0f5', '#d2aaf0','#d8b3e6', '#9467bd', '#e377c2', '#17becf', 'violet', 'pink', 'olive', 'gold', 'goldenrod',
                          'antiquewhite', '#ffcccc', '#ff6666', '#ff0000', '#b30000', '#800000']

            for idx, biopsy_list in enumerate(biopsy_lists):
                color = red_shades[idx % len(red_shades)]  # Loop through shades of red if there are more than 5 lists

                # Collect the nodes to highlight based on biopsy cell_id matches
                for node, data in G.nodes(data=True):
                    cell_id = data.get("cell_id", "N/A")
                    # If the cell_id matches any in the current biopsy list, add the coordinates and color
                    if cell_id in [gen.cell_id for gen in biopsy_list]:
                        x, y = scaled_pos[node]
                        biopsy_highlight_x.append(x)
                        biopsy_highlight_y.append(y)
                        biopsy_colors.append(color)

        # Edge and node plotting
        edge_x, edge_y = [], []
        edge_labels = []
        edge_label_pos_x, edge_label_pos_y = [], []
        hover_edge_labels = []
        edge_marker_colors = []
        edge_positions = {}  # Store midpoint for annotations

        for (src, tgt), event in nx.get_edge_attributes(G, 'events').items():
            x0, y0 = scaled_pos[src]
            x1, y1 = scaled_pos[tgt]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            edge_label_pos_x.append(mid_x)
            edge_label_pos_y.append(mid_y)
            edge_labels.append("")  # Hide label by default
            hover_edge_labels.append(str(event))  # Show label only on hover
            edge_positions[(src, tgt)] = (mid_x, mid_y)

            edge_marker_colors.append("red" if event else "green")  # Red if label exists, green otherwise

        # Create Plotly figure
        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=1, color='black'),
            hoverinfo='none',
            showlegend=False
        ))

        if only_nodes is not None:
            important = []
            for node, data in G.nodes(data=True):
                cell_id = data.get("cell_id", "N/A")
                for biopsy_list in biopsy_lists:
                    if cell_id in [gen.cell_id for gen in biopsy_list]:
                        important.append(node)
            only_nodes += important

            # Filter nodes based on only_nodes
            visible_node_x = []
            visible_node_y = []
            visible_hover_labels = []
            highlight_texts = []

            for x, y, hover, node in zip(node_x, node_y, hover_labels, G.nodes()):
                if node in only_nodes:
                    visible_node_x.append(x)
                    visible_node_y.append(y)
                    visible_hover_labels.append(hover)
                    cell_id = G.nodes[node].get("cell_id", "")
                    highlight_texts.append(str(cell_id))

            # Add nodes (excluding those not in only_nodes)
            fig.add_trace(go.Scatter(
                x=visible_node_x, y=visible_node_y, mode='markers',
                marker=dict(size=10, color='lightblue', line=dict(width=1, color='black')),
                hovertext=visible_hover_labels,
                name="cancer cell",
                hoverinfo='text',
                showlegend=True
            ))
            if node_numbers:
                # Add nearby text labels for cell_id
                fig.add_trace(go.Scatter(
                    x=[x + 25 for x in visible_node_x],  # slight horizontal offset
                    y=visible_node_y,
                    text=highlight_texts,
                    mode='text',
                    textposition='middle right',
                   textfont=dict(size=14, color='purple'),
                   showlegend=True,
                    name="cell genotype id",
                    hoverinfo='skip'
                ))
        else:
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y, mode='markers+text',
                marker=dict(size=10, color='lightblue', line=dict(width=1, color='black')),
                text=node_labels, textposition='top center',
                showlegend=True,
                hoverinfo='text',
                name="cancer cell",
                hovertext=hover_labels  # Full info on hover
            ))

        if extended:
            # Add markers for edge labels
            fig.add_trace(go.Scatter(
                x=edge_label_pos_x, y=edge_label_pos_y, mode='markers+text',
                marker=dict(size=8, color=edge_marker_colors, opacity=0.5),  # Change color based on label presence
                text=edge_labels,
                hovertext=hover_edge_labels,  # Show edge label on hover
                textposition='middle center',
                hoverinfo='text',
                showlegend=False
            ))

            # Add green edge markers
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=8, color='green', opacity=0.5),
                name="no mutational events", hoverinfo='skip',
                showlegend=True,
            ))

            # Add red edge markers
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=8, color='red', opacity=0.5),
                name="mutational event occurrence", hoverinfo='skip',
                showlegend=True,
            ))

        # Add biopsy highlights if any
        if biopsy_lists is not None:
            fig.add_trace(go.Scatter(
                x=biopsy_highlight_x, y=biopsy_highlight_y, mode='markers',
                marker=dict(size=12, color=biopsy_colors, symbol='circle', line=dict(width=2, color='black')),
                showlegend=True,
                name="cells sharing the same genotype as the biopsy cells",
                hoverinfo='none'
            ))

        # Add blue circles for highlighted nodes
        if highlight_nodes:
            highlight_circle_x = []
            highlight_circle_y = []
            highlight_texts = []

            for n in highlight_nodes:
                node_id = n.node_id
                if node_id in scaled_pos:
                    x, y = scaled_pos[node_id]
                    highlight_circle_x.append(x)
                    highlight_circle_y.append(y)
                    # Add cell_id text if available
                    cell_id = G.nodes[node_id].get("cell_id", "")
                    highlight_texts.append(str(cell_id))

            fig.add_trace(go.Scatter(
                x=highlight_circle_x, y=highlight_circle_y,
                mode='markers',
                marker=dict(
                    size=20,
                    color='rgba(0,0,0,0)',
                    line=dict(color='purple', width=3)   #  darkcyan, purple, red, crimson, deepskyblue
                ),
                showlegend=True,
                name="cells selected by biopsy",
                hoverinfo='skip'
            ))
            # if node_numbers:
            #     # Add nearby text labels for cell_id
            #     fig.add_trace(go.Scatter(
            #         x=[x + 25 for x in highlight_circle_x],  # slight horizontal offset
            #         y=highlight_circle_y,
            #         text=highlight_texts,
            #         mode='text',
            #         textposition='middle right',
            #        textfont=dict(size=14, color='purple'),
            #        showlegend=False,
            #         hoverinfo='skip'
            #     ))

        # Determine most-left and topmost node position for legend anchor
        min_x_val = min(node_x)
        max_x_val = max(node_x)
        min_y_val = min(node_y)
        max_y_val = max(node_y)
        rel_x = (min_x_val + legend_x_offset) / (max_x_val - min_x_val)
        rel_y = (max_y_val + legend_y_offset) / (max_y_val - min_y_val)
        # Final layout
        fig.update_layout(
            title="",
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            legend=dict(
                title="",
                x=rel_x,  # move to the right
                y=rel_y,
                xanchor='left', yanchor='top',
                bgcolor="rgba(255,255,255,0.3)",
                bordercolor="black",
                borderwidth=0,
                font=dict(size=18)
            )
        )

        # Save and show
        if output_file is not None:
            fig.write_image(output_file+".png", width=1200, height=800, scale=2)
            fig.write_html(output_file+".html")
            fig.write_image(output_file + ".svg", width=1200, height=800, scale=2)
        fig.show()
