import networkx as nx
import itertools
import re
from collections import Counter


GRF_METRIC_NAME = "grf"
GRF_METRIC_KIND = "similarity"
GRF_HIGHER_IS_BETTER = True
GRF_SCORE_RANGE = (0.0, 1.0)
GRF_METRIC_DESCRIPTION = (
    "Cluster-multiset topology similarity. 1.0 means identical cluster structure "
    "under this score; lower values mean poorer agreement. This is not a distance."
)
EXT_GRF_METRIC_NAME = "ext_grf"
EXT_GRF_METRIC_KIND = "distance"
EXT_GRF_HIGHER_IS_BETTER = False
EXT_GRF_SCORE_RANGE = (0.0, 1.0)
EXT_GRF_METRIC_DESCRIPTION = (
    "Exact generalized Robinson-Foulds distance on multisets of label multisets. "
    "0.0 means identical cluster-multiset structure; higher values mean poorer agreement."
)

def parse_newick_to_nx(newick_str, prefix="node"):
    """
    Parses a Newick string into a NetworkX DiGraph.
    Returns the graph and the root node ID.
    """
    G = nx.DiGraph()
    stack = []
    uid_counter = itertools.count()

    tokens = re.findall(r'\(|\)|,|;|[^(),;]+', newick_str)
    idx = 0
    current_node_id = None

    while idx < len(tokens):
        token = tokens[idx]
        if token == '(':
            stack.append([])
            idx += 1
        elif token == ')':
            children = stack.pop()
            idx += 1
            label = None
            if idx < len(tokens) and re.match(r'[^(),;]+', tokens[idx]):
                label = tokens[idx]
                idx += 1
            node_id = f"{prefix}_{next(uid_counter)}"
            G.add_node(node_id, cell_id=label)
            for child in children:
                G.add_edge(node_id, child)
            if stack:
                stack[-1].append(node_id)
            else:
                current_node_id = node_id
        elif token == ',':
            idx += 1
        elif token == ';':
            idx += 1
        else:
            label = token
            node_id = f"{prefix}_{next(uid_counter)}"
            G.add_node(node_id, cell_id=label)
            if stack:
                stack[-1].append(node_id)
            else:
                current_node_id = node_id
            idx += 1

    return G, current_node_id

def compute_all_clusters(G, root):
    """
    Computes the multiset of labels for all subtrees in a single post-order traversal.
    Uses sorted tuples for faster comparison and hashing.
    """
    clusters = {}

    def dfs(n):
        counter = Counter()
        cell_id = G.nodes[n]['cell_id']
        if cell_id is not None:
            counter[cell_id] += 1
        for child in G.successors(n):
            counter += dfs(child)
        cluster = tuple(sorted(counter.items()))
        clusters[n] = cluster
        return counter

    dfs(root)
    return list(clusters.values())

def jaccard_distance(ms1, ms2):
    """
    Computes Jaccard distance using two sorted (label, count) tuples.
    """
    i = j = 0
    intersection = union = 0

    while i < len(ms1) and j < len(ms2):
        label1, count1 = ms1[i]
        label2, count2 = ms2[j]
        if label1 == label2:
            intersection += min(count1, count2)
            union += max(count1, count2)
            i += 1
            j += 1
        elif label1 < label2:
            union += count1
            i += 1
        else:
            union += count2
            j += 1

    while i < len(ms1):
        union += ms1[i][1]
        i += 1
    while j < len(ms2):
        union += ms2[j][1]
        j += 1

    return 1 - (intersection / union) if union else 0


def _multiset_union_size(counts1, counts2):
    return sum(
        max(counts1.get(item, 0), counts2.get(item, 0))
        for item in counts1.keys() | counts2.keys()
    )


def _multiset_difference_counts(left_counts, right_counts):
    return Counter(
        {
            item: count - right_counts.get(item, 0)
            for item, count in left_counts.items()
            if count > right_counts.get(item, 0)
        }
    )


def _weighted_jaccard_distance_sum(left_counts, right_counts):
    return sum(
        left_multiplicity
        * right_multiplicity
        * jaccard_distance(left_cluster, right_cluster)
        for left_cluster, left_multiplicity in left_counts.items()
        for right_cluster, right_multiplicity in right_counts.items()
    )


def _ext_grf_from_clusters(A, B):
    A_counts = Counter(A)
    B_counts = Counter(B)
    len_A = sum(A_counts.values())
    len_B = sum(B_counts.values())
    union_size = _multiset_union_size(A_counts, B_counts)

    if union_size == 0:
        return 0.0
    if len_A == 0 or len_B == 0:
        return 1.0

    b_minus_a = _multiset_difference_counts(B_counts, A_counts)
    a_minus_b = _multiset_difference_counts(A_counts, B_counts)

    num1 = _weighted_jaccard_distance_sum(A_counts, b_minus_a)
    num2 = _weighted_jaccard_distance_sum(a_minus_b, B_counts)

    return (num1 / (len_A * union_size)) + (num2 / (len_B * union_size))


def ext_grf_tree(G1, root1, G2, root2):
    """
    Return the exact generalized Robinson-Foulds distance.

    Directionality: lower is better. The implementation follows the literature
    formula over multisets of clusters, so repeated equal clusters contribute
    with their multiplicity instead of being collapsed to set membership.
    """
    A = compute_all_clusters(G1, root1)
    B = compute_all_clusters(G2, root2)
    return _ext_grf_from_clusters(A, B)


def grf_tree(G1, root1, G2, root2):
    """
    Return the project-facing generalized Robinson-Foulds similarity score.

    Directionality: higher is better. This is retained as the legacy benchmark
    value and is defined as 1 minus the exact GRF distance returned by
    `ext_grf_tree`.
    """
    return 1 - ext_grf_tree(G1, root1, G2, root2)


def ext_grf(newick1, newick2):
    """Return the exact generalized Robinson-Foulds distance for two Newick strings."""
    G1, root1 = parse_newick_to_nx(newick1, prefix="A")
    G2, root2 = parse_newick_to_nx(newick2, prefix="B")
    return ext_grf_tree(G1, root1, G2, root2)

def grf(newick1, newick2):
    """Return the project-facing GRF similarity for two Newick strings."""
    G1, root1 = parse_newick_to_nx(newick1, prefix="A")
    G2, root2 = parse_newick_to_nx(newick2, prefix="B")
    return grf_tree(G1, root1, G2, root2)

def rgrf(newick1, newick2):
    G1, root1 = parse_newick_to_nx(newick1, prefix="A")
    G2, root2 = parse_newick_to_nx(newick2, prefix="B")
    return rgrf_tree(G1, root1, G2, root2)

def rgrf_tree(G1, root1, G2, root2):
    # Determine leaf and shared internal cell_ids
    def get_leaf_cell_ids(G):
        return {
            G.nodes[n]['cell_id']
            for n in G.nodes
            if G.out_degree(n) == 0 and G.nodes[n]['cell_id'] is not None
        }

    def get_internal_cell_ids(G):
        return {
            G.nodes[n]['cell_id']
            for n in G.nodes
            if G.out_degree(n) > 0 and G.nodes[n]['cell_id'] is not None
        }

    leaf_ids_A = get_leaf_cell_ids(G1)
    leaf_ids_B = get_leaf_cell_ids(G2)
    internal_ids_A = get_internal_cell_ids(G1)
    internal_ids_B = get_internal_cell_ids(G2)

    allowed_labels = (leaf_ids_A | leaf_ids_B) | (internal_ids_A & internal_ids_B)
    print(allowed_labels)

    def filtered_get_label_multiset(G, node_id, allowed_labels):
        counter = Counter()

        def dfs(n):
            cell_id = G.nodes[n]['cell_id']
            if cell_id is not None and cell_id in allowed_labels:
                counter[cell_id] += 1
            for child in G.successors(n):
                dfs(child)

        dfs(node_id)
        return frozenset(counter.items())

    def filtered_tree_to_clusters(G, root, allowed_labels):
        clusters = []

        def dfs(n):
            cluster = filtered_get_label_multiset(G, n, allowed_labels)
            clusters.append(cluster)
            for child in G.successors(n):
                dfs(child)

        dfs(root)
        return clusters

    A = filtered_tree_to_clusters(G1, root1, allowed_labels)
    B = filtered_tree_to_clusters(G2, root2, allowed_labels)
    A_set, B_set = set(A), set(B)

    union_size = len(A_set | B_set)
    if union_size == 0:
        return 0.0

    num1 = sum(jaccard_distance(a, b) for a in A for b in B if b not in A_set)
    num2 = sum(jaccard_distance(b, a) for b in B for a in A if a not in B_set)

    return 1 - ((num1 / (len(A) * union_size)) + (num2 / (len(B) * union_size)))

def bgrf(newick1, newick2, allowed_labels):
    G1, root1 = parse_newick_to_nx(newick1, prefix="A")
    G2, root2 = parse_newick_to_nx(newick2, prefix="B")
    return bgrf_tree(G1, root1, G2, root2, allowed_labels)


def bgrf_tree(G1, root1, G2, root2, allowed_labels):

    def filtered_get_label_multiset(G, node_id, allowed_labels):
        counter = Counter()

        def dfs(n):
            cell_id = G.nodes[n]['cell_id']
            if cell_id is not None and cell_id in allowed_labels:
                counter[cell_id] += 1
            for child in G.successors(n):
                dfs(child)

        dfs(node_id)
        return frozenset(counter.items())

    def filtered_tree_to_clusters(G, root, allowed_labels):
        clusters = []

        def dfs(n):
            cluster = filtered_get_label_multiset(G, n, allowed_labels)
            clusters.append(cluster)
            for child in G.successors(n):
                dfs(child)

        dfs(root)
        return clusters

    A = filtered_tree_to_clusters(G1, root1, allowed_labels)
    B = filtered_tree_to_clusters(G2, root2, allowed_labels)
    A_set, B_set = set(A), set(B)

    union_size = len(A_set | B_set)
    if union_size == 0:
        return 0.0

    A = [x for x in A_set if x != set()]
    B = [x for x in A_set if x != set()]
    num1 = sum(jaccard_distance(a, b) for a in A for b in B if b not in A_set)
    num2 = sum(jaccard_distance(b, a) for b in B for a in A if a not in B_set)

    return 1 - ((num1 / (len(A) * union_size)) + (num2 / (len(B) * union_size)))

if __name__ == "__main__":
    tree_A = "((a,b)c,(d)d)c;"
    tree_B = "((a,b),(d)d);"

    tree_A, tree_B = "((a,b)e,(d)f)g;", "(e,f)g;"
    tree_A, tree_B = "((((a,b)d,e)f,h)i,c)g;", "((a,b),c);"
    # tree, root = parse_newick_to_nx(tree_B)
    # print([data['cell_id'] for x, data in tree.nodes(data=True)])

    print("GRF similarity:", grf(tree_A, tree_B))
    print("ext_GRF distance:", ext_grf(tree_A, tree_B))
    # print("RGRF distance:", rgrf(tree_A, tree_B))
    # print("BGRF distance:", bgrf(tree_A, tree_B, {'a', 'b', 'c'}))

    print(grf("((a,b)f,(c,d)e)g;", "((a,b)f,c);"))
