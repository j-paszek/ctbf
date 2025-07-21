import json
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import csv

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

def load_genome_from_csv(csv_path):
    genome = []
    event_probs = []
    duplication_probs = []
    duplication_multiplicities = []
    loss_probs = []
    single_or_multiple_probs = []
    telomeric_instabilities = []
    chromosomes = []
    crucial_for_survival = []
    fitness_weights = []  # Add fitness_weights list

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)  # Using DictReader for easier column access by name
        for row in reader:
            # Split the 'Parameters' column into key-value pairs and process them
            params = {param.split('=')[0]: float(param.split('=')[1]) for param in row['Parameters'].split(';') if
                      '=' in param}

            # Extract chromosome, CN, and other parameters
            if 'ChromosomeNumber' in row:
                chromosome = int(row['ChromosomeNumber'])
                chromosomes.append(chromosome)

            cn = int(params.get("CN", 2))  # Default CN to 2 if not present in parameters
            fitness_weight = float(params.get("FITNESS_WEIGHT", 0))  # Default FITNESS_WEIGHT to 0 if missing

            genome.append(cn)
            event_probs.append(params.get("EVENT_PROB", None))
            duplication_probs.append(params.get("DUPLICATION_PROB", None))
            duplication_multiplicities.append(params.get("DUPLICATION_MULTIPLICITY", None))
            loss_probs.append(params.get("LOSS_PROB", None))
            single_or_multiple_probs.append(params.get("SINGLE_OR_MULTIPLE_EVENT_PROB", None))
            telomeric_instabilities.append(params.get("TELOMERIC_INSTABILITY", None))
            crucial_for_survival.append(params.get("CRUCIAL", 0))
            fitness_weights.append(fitness_weight)  # Store fitness weight

    if all(x == 0 for x in fitness_weights):
        fitness_weights = None

    if all(x == 0 for x in crucial_for_survival):
        crucial_for_survival = None

    return (genome, event_probs, duplication_probs, duplication_multiplicities, loss_probs,
            single_or_multiple_probs, telomeric_instabilities, chromosomes, crucial_for_survival, fitness_weights)


"""
A simulator for cell population evolution using a replicator model.
Simulation parameters (loaded from config.json) define:
  - genome length, initial copy number, number of generations
  - CN event parameters (expected_events, expected_event_length, duplication_rate, position_probabilities)
  - offspring count distribution (e.g., Poisson)
"""


class CancerCellEvolutionSimulator:
    def __init__(self, config_path, genome_csv=None, seed=None):
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Seed for reproducibility
        self.seed = seed
        # self.chidi = {}
        if self.seed is not None:
            np.random.seed(self.seed)

        self.model_telomeric_regions = str(config.get("MODEL_TELOMERIC_REGIONS", "False")).lower() == "true"
        self.model_crucial_for_survival = str(config.get("MODEL_CRUCIAL_FOR_SURVIVAL", "False")).lower() == "true"

        if self.model_telomeric_regions:
            if "GENERAL_TELOMERIC_INSTABILITY" not in config or "GENERAL_TELOMERIC_PERCENTAGE" not in config:
                raise ValueError(
                    "GENERAL_TELOMERIC_INSTABILITY and GENERAL_TELOMERIC_PERCENTAGE must be specified in the config when MODEL_TELOMERIC_REGIONS is enabled.")
            self.general_telomeric_instability = config["GENERAL_TELOMERIC_INSTABILITY"]
            self.general_telomeric_percentage = config["GENERAL_TELOMERIC_PERCENTAGE"]

        # generate or load founder_genome
        founder_genome = []
        if genome_csv is not None:
            (self.genome, self.event_probs, self.duplication_probs,
             self.duplication_multiplicities, self.loss_probs,
             self.single_or_multiple_probs, self.telomeric_instabilities,
             chromosome, crucial_for_survival, fitness_weights) = load_genome_from_csv(genome_csv)
            if chromosome:
                self.chromosome = chromosome
            if crucial_for_survival:
                self.crucial_for_survival = crucial_for_survival
            if fitness_weights:
                self.fitness_weights = fitness_weights
            self.genome_length = len(self.genome)
            founder_genome = np.array(self.genome)
        else:
            self.genome_length = config["genome_length"]
            self.initial_copies = config["initial_copies"]
            founder_genome = np.full(self.genome_length, self.initial_copies)

        self.founder_genome = founder_genome

        if self.model_telomeric_regions:
            self.telomeric_instability_factor = np.zeros(self.genome_length)

            if not hasattr(self, 'chromosome') or self.chromosome is None:
                telomeric_size = int(len(self.founder_genome) * self.general_telomeric_percentage)
                # Only apply the instability if telomeric_size is greater than 0
                if telomeric_size > 0:
                    # Assign the general telomeric instability value to the first and last portions of this chromosome
                    self.telomeric_instability_factor[:telomeric_size] = self.general_telomeric_instability
                    self.telomeric_instability_factor[-telomeric_size:] = self.general_telomeric_instability

                # Overwrite with specific values from the CSV if provided
                for i in range(len(self.founder_genome)):
                    if self._get_ith_telomeric_instability(i) is not None:
                        self.telomeric_instability_factor[i] = self.telomeric_instabilities[i]

            else:
                # Iterate over distinct chromosomes
                unique_chromosomes = np.unique(self.chromosome)  # Get unique chromosome numbers
                for chrom in unique_chromosomes:
                    # Get the indices of the positions corresponding to this chromosome
                    chrom_indices = np.where(self.chromosome == chrom)[0]
                    chrom_length = len(chrom_indices)

                    # Calculate the telomeric size for this chromosome
                    telomeric_size = int(chrom_length * self.general_telomeric_percentage)

                    # Only apply the instability if telomeric_size is greater than 0
                    if telomeric_size > 0:
                        # Assign the general telomeric instability value to the first and last portions of this chromosome
                        self.telomeric_instability_factor[
                            chrom_indices[:telomeric_size]] = self.general_telomeric_instability
                        self.telomeric_instability_factor[
                            chrom_indices[-telomeric_size:]] = self.general_telomeric_instability

                    # Overwrite with specific values from the CSV if provided
                    for i in chrom_indices:
                        if self.telomeric_instabilities[i] is not None:
                            self.telomeric_instability_factor[i] = self.telomeric_instabilities[i]

        # Simulation parameters
        self.num_generations = config["NUMBER_OF_GENERATIONS"]
        self.offspring_model = config["OFFSPRING_MODEL"]
        self.offspring_param = config["OFFSPRING_PARAMETER"]

        self.event_prob = config["GENERAL_EVENT_PROB"]
        self.duplication_prob = config["GENERAL_DUPLICATION_PROB"]
        self.duplication_multiplicity = config["GENERAL_DUPLICATION_MULTIPLICITY"]
        self.loss_prob = config["GENERAL_LOSS_PROB"]
        self.single_or_multiple_event_prob = config["GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB"]

        self.representation_type = config.get("REPRESENTATION_TYPE", "representative")
        if self.representation_type not in {"full", "representative"}:
            raise ValueError("REPRESENTATION_TYPE must be 'full' or 'representative'")

        # Evolutionary tree initialization
        self.tree = nx.DiGraph()
        # Population initialization: start with a single genotype (founder)
        founder = Genotype(genome=founder_genome, node_id=0, generation=0)
        # node_id -> Genotype
        self.genotypes = {0: founder}

        # Add founder node to the graph
        self.tree.add_node(0, genome=founder_genome.tolist(), generation=0, cell_id=0)
        self.node_counter = 1

    def get_parameters_csv(self, file):
        """Outputs a CSV file containing CN values, event probabilities, and crucial_for_survival status."""
        with open(file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ["CN", "Chromosome", "GENERAL_EVENT_PROB", "GENERAL_DUPLICATION_PROB",
                       "GENERAL_DUPLICATION_MULTIPLICITY", "GENERAL_LOSS_PROB", "GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB",
                       "CRUCIAL_FOR_SURVIVAL", "FITNESS_WEIGHT"]
            if self.model_telomeric_regions:
                headers.append("TELOMERIC_INSTABILITY")
            writer.writerow(headers)
            for i, cn in enumerate(self.founder_genome):
                row = [cn, self._get_ith_chromosome(i), self._get_ith_event_probability(i),
                       self._get_ith_duplication_probability(i), self._get_ith_duplication_multiplicity(i),
                       self._get_ith_loss_probability(i), self._get_ith_single_or_multiple_probability(i),
                       self._get_ith_crucial(i), self._get_ith_fitness(i)]  # Include FITNESS_WEIGHT
                if self.model_telomeric_regions:
                    row.append(self.telomeric_instability_factor[i])
                writer.writerow(row)

    def create_bed_csv(self, file):
        with (open(file, mode='w', newline='') as csvfile):
            writer = csv.writer(csvfile)
            headers = ["ChromosomeNumber","Start","End","Parameters"]
            writer.writerow(headers)
            for i, cn in enumerate(self.founder_genome):
                c = "CN="+str(cn)
                row = [1,0,0,c]
                writer.writerow(row)

    def _sample_offspring_count(self, genotype):
        if self.offspring_model == "constant":
            return int(self.offspring_param)
        elif self.offspring_model == "uniform_range":
            min_val, max_val = map(int, self.offspring_param.split(","))
            return np.random.randint(min_val, max_val + 1)
        elif self.offspring_model == "poisson":
            return np.random.poisson(float(self.offspring_param))
        elif self.offspring_model == "fitness":
            return np.random.poisson(self._fitness_poisson_mean(genotype, MAX_N=self.offspring_param))
            # total_fitness = self._compute_fitness(genotype)
            # # print(total_fitness)
            # exp_children = total_fitness + 2
            # if exp_children < 0:
            #     exp_children = 0
            # if exp_children > 4:
            #     exp_children = 4
            # return np.random.poisson(exp_children)
        else:
            raise ValueError("Unsupported OFFSPRING_MODEL.")

    def _fitness_poisson_mean(self, genotype, alpha=1.0, MAX_N=4.0):
        """
        Computes the expected number of offspring (Poisson mean)
        based on a logistic transform of the dot-product of (cnp - 1) and weights.

        Parameters:
        -----------
        cnp     : array-like of copy numbers, shape (n,)
        importance_weights : array-like of fitness impacts, shape (n,)
        alpha   : float, shape parameter controlling how sharply
                  fitness rises or falls around baseline
        MAX_N   : What is the maximum number of offspring?
        Returns:
        --------
        float    : The mean (lambda) of the Poisson distribution
                   for the expected number of offspring.
        """
        # Convert inputs to arrays, just to be sure
        cnp = np.array(genotype.genome)
        importance_weights = np.zeros(len(genotype.genome))
        for i in range(len(genotype.genome)):
            importance_weights[i] = self._get_ith_fitness(i)

        # Score S = sum_i w_i * (cn_i - 1)
        score = np.sum(importance_weights * (cnp - 1))
        # Logistic transform bounding output to (0, 4)
        # This ensures baseline genotype (cnp = 1,...,1) => score=0 => output=2
        numerator = MAX_N * np.exp(alpha * score)
        denominator = 1.0 + np.exp(alpha * score)
        lam = numerator / denominator

        return lam

    def _compute_fitness(self, genotype):
        # Sum FITNESS_WEIGHT * CN for each position
        suma = 0
        for i in range(self.genome_length):
            suma += self._get_ith_fitness(i) * genotype.genome[i]
        return suma / self.genome_length

    def _get_ith_fitness(self, i):
        return (self.fitness_weights[i]
                if hasattr(self, 'fitness_weights') and self.fitness_weights and
                   self.fitness_weights[i] is not None
                else 0)

    def _get_ith_crucial(self, i):
        return (self.crucial_for_survival[i]
                if hasattr(self, 'crucial_for_survival') and self.crucial_for_survival and self.crucial_for_survival[
            i] is not None
                else 0)

    def _get_ith_telomeric_instability(self, i):
        return (self.telomeric_instabilities[i]
                if hasattr(self, 'telomeric_instabilities') and self.telomeric_instabilities and
                   self.telomeric_instabilities[i] is not None
                else None)

    def _get_ith_chromosome(self, i):
        return (self.chromosome[i]
                if hasattr(self, 'chromosome') and self.chromosome and self.chromosome[i] is not None
                else 1)

    def _get_ith_event_probability(self, i):
        return (self.event_probs[i]
                if hasattr(self, 'event_probs') and self.event_probs and self.event_probs[i] is not None
                else self.event_prob)

    def _get_ith_duplication_probability(self, i):
        return (self.duplication_probs[i]
                if hasattr(self, 'duplication_probs') and self.duplication_probs and self.duplication_probs[
            i] is not None
                else self.duplication_prob)

    def _get_ith_duplication_multiplicity(self, i):
        return (self.duplication_multiplicities[i]
                if hasattr(self, 'duplication_multiplicities') and self.duplication_multiplicities and
                   self.duplication_multiplicities[i] is not None
                else self.duplication_multiplicity)

    def _get_ith_loss_probability(self, i):
        return (self.loss_probs[i]
                if hasattr(self, 'loss_probs') and self.loss_probs and self.loss_probs[i] is not None
                else self.loss_prob)

    def _get_ith_single_or_multiple_probability(self, i):
        return (self.single_or_multiple_probs[i]
                if hasattr(self, 'single_or_multiple_probs') and self.single_or_multiple_probs and
                   self.single_or_multiple_probs[i] is not None
                else self.single_or_multiple_event_prob)

    def _apply_copy_number_events(self, genome):
        events_summary = []
        for i in range(self.genome_length):
            event_prob = self._get_ith_event_probability(i)
            if self.model_telomeric_regions:
                event_prob = min(1, event_prob + self.telomeric_instability_factor[i])

            # event_prob = self.event_probs[i] if self.event_probs[i] is not None else self.event_prob
            if np.random.rand() < event_prob:
                is_multiple_event = np.random.rand() < self.single_or_multiple_event_prob
                if is_multiple_event:
                    j = np.random.randint(i, self.genome_length)
                else:
                    j = i

                event_type = "duplication" if np.random.rand() < self.duplication_prob else "loss"
                num_copies = np.random.poisson(self.duplication_multiplicity) + 1 if event_type == "duplication" else -1

                for pos in range(i, j + 1):
                    genome[pos] = max(0, genome[pos] + num_copies)
                    if self.model_crucial_for_survival and self._get_ith_crucial(i) and genome[pos] == 0:
                        return None, None  # Prevent genome generation if crucial CN drops to 0
                pos_label = f"{i}-{j}" if i != j else f"{i}"
                events_summary.append(f"{event_type}(pos={pos_label}, copies={num_copies})")
        return genome, ";".join(events_summary)

    """
    Run the simulation for a specified number of generations.
    In each generation:
      1. Generate offspring for existing genotypes.
    """

    def run_simulation(self):
        for gen in range(1, self.num_generations + 1):
            self._spawn_children(current_generation=gen)

    """
    For each genotype, draw the number of offspring based on Poisson distribution.
    For each child, create a new genome by copying the parent and applying mutations.
    """

    def _spawn_children(self, current_generation):
        new_genotypes = []
        parent_genotypes = [g for g in self.genotypes.values() if g.generation == current_generation - 1]
        seen_genomes = set()
        for genotype in parent_genotypes:
            # Randomly generate the number of offspring for this genotype
            num_children = self._sample_offspring_count(genotype)
            # if num_children in self.chidi:
            #     self.chidi[num_children] += 1
            # else:
            #     self.chidi[num_children] = 1
            for _ in range(num_children + 1):
                # Create a copy of the parent's genome
                child_genome, event_summary = self._apply_copy_number_events(genotype.genome.copy())

                # Ignore the child if crucial CN is lost (genome is None)
                if child_genome is None:
                    continue

                if self.representation_type == "representative":
                    genome_tuple = tuple(child_genome)
                    if genome_tuple in seen_genomes:
                        continue
                    seen_genomes.add(genome_tuple)
                # Create a new genotype
                if event_summary:
                    child_cell_id = self.node_counter
                else:
                    child_cell_id = genotype.cell_id
                child = Genotype(
                    genome=child_genome,
                    node_id=self.node_counter,
                    generation=current_generation,
                    cell_id=child_cell_id
                )
                # Store information about the parent
                new_genotypes.append((genotype.node_id, child, event_summary))
                # Add a node to the tree
                self.tree.add_node(
                    self.node_counter,
                    genome=child_genome.tolist(),
                    generation=current_generation,
                    cell_id=child_cell_id
                )
                # Create edge from parent -> child with event annotation
                self.tree.add_edge(
                    genotype.node_id,
                    self.node_counter,
                    events=event_summary
                )
                self.node_counter += 1

        # Add newly created genotypes to the dictionary
        for parent_id, child, event_summary in new_genotypes:
            self.genotypes[child.node_id] = child


    def perform_biopsy(self, generation, biopsy_size=0, biopsy_size_scalable=None, seed=None):
        """
        Randomly selects unique genotypes from a given generation.
        Returns a list of selected genotype objects.
        """
        # Get genotypes from the specified generation
        genotypes_from_generation = [g for g in self.genotypes.values() if g.generation == generation]

        # If there are fewer genotypes than the biopsy size, return all available
        biopsy_size = min(biopsy_size, len(genotypes_from_generation))
        if biopsy_size_scalable is not None:
            biopsy_size = int(biopsy_size_scalable * len(genotypes_from_generation))

        if seed is not None:
            np.random.seed(seed)
        # Randomly sample unique genotypes
        sampled_genotypes = np.random.choice(genotypes_from_generation, size=biopsy_size, replace=False)

        return list(sampled_genotypes)

    def tree_without_CNPs(self):
        """Return a copy of the tree where each node's genome is replaced with None to free memory."""
        tree_copy = self.tree.copy()
        for node in tree_copy.nodes:
            tree_copy.nodes[node]['genome'] = None
        return tree_copy

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
        fig.show()