import networkx as nx
import itertools
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from evaluator_full import normalize_cell_label


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


@dataclass(frozen=True)
class ClusterEvaluationContext:
    clusters: tuple
    counts: Counter
    cluster_set: frozenset


@dataclass(frozen=True)
class _CompactClusterView:
    items: tuple
    counts: dict
    size: int


@dataclass(frozen=True)
class ClusterComparisonWork:
    label_to_id: dict
    view_cache: dict


@dataclass(frozen=True)
class GrfComparisonMetadata:
    left_size: int
    right_size: int
    multiset_union_size: int
    left_minus_right: Counter
    right_minus_left: Counter
    set_union_size: int
    left_only_counts: Counter
    right_only_counts: Counter


_DENSE_PAIR_THRESHOLD = 8192
_DENSE_LABEL_LIMIT = 512
_DENSE_WORK_LIMIT = 20_000_000


def _looks_like_tree_context(value):
    return all(hasattr(value, name) for name in ("labels", "children", "roots"))

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
        cell_id = normalize_cell_label(G.nodes[n].get("cell_id"))
        if cell_id is not None:
            counter[cell_id] += 1
        for child in G.successors(n):
            counter += dfs(child)
        cluster = tuple(sorted(counter.items()))
        clusters[n] = cluster
        return counter

    dfs(root)
    return list(clusters.values())


def _single_context_root(context, root=None):
    if root is not None:
        return root
    roots = tuple(context.roots)
    if len(roots) != 1:
        raise ValueError(f"Tree must have exactly one root, found {len(roots)}")
    return roots[0]


def compute_all_clusters_from_context(context, root=None):
    root = _single_context_root(context, root)
    clusters = []
    counters = {}
    stack = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            counter = Counter()
            child_counters = [
                counters.pop(child)
                for child in context.children.get(node, ())
                if child in counters
            ]
            if child_counters:
                largest_index = max(
                    range(len(child_counters)),
                    key=lambda index: len(child_counters[index]),
                )
                if counter:
                    counter += child_counters.pop(largest_index)
                else:
                    counter = child_counters.pop(largest_index)
            for child_counter in child_counters:
                counter += child_counter
            label = context.labels.get(node)
            if label is not None:
                counter[label] += 1
            cluster = tuple(sorted(counter.items()))
            clusters.append(cluster)
            counters[node] = counter
            continue
        stack.append((node, True))
        for child in reversed(context.children.get(node, ())):
            stack.append((child, False))
    return clusters


def cluster_evaluation_context(G, root=None):
    if _looks_like_tree_context(G):
        clusters = tuple(compute_all_clusters_from_context(G, root))
    elif isinstance(G, dict) and "nodes" in G:
        from evaluator_full import tree_evaluation_context_from_node_link

        context = tree_evaluation_context_from_node_link(G)
        clusters = tuple(compute_all_clusters_from_context(context, root))
    else:
        clusters = tuple(compute_all_clusters(G, root))
    return ClusterEvaluationContext(
        clusters=clusters,
        counts=Counter(clusters),
        cluster_set=frozenset(clusters),
    )


def _cached_jaccard_distance(left_cluster, right_cluster, cache):
    if cache is None:
        return jaccard_distance(left_cluster, right_cluster)

    key = (left_cluster, right_cluster)
    if key in cache:
        return cache[key]
    reverse_key = (right_cluster, left_cluster)
    if reverse_key in cache:
        return cache[reverse_key]
    distance = jaccard_distance(left_cluster, right_cluster)
    cache[key] = distance
    return distance


def _intern_cluster_label(label, label_to_id):
    label_id = label_to_id.get(label)
    if label_id is not None:
        return label_id
    label_id = len(label_to_id)
    label_to_id[label] = label_id
    return label_id


def _compact_cluster_view(cluster, label_to_id, view_cache):
    view = view_cache.get(cluster)
    if view is not None:
        return view

    counts = {}
    size = 0
    for label, count in cluster:
        label_id = _intern_cluster_label(label, label_to_id)
        counts[label_id] = counts.get(label_id, 0) + count
        size += count
    view = _CompactClusterView(
        items=tuple(sorted(counts.items())),
        counts=counts,
        size=size,
    )
    view_cache[cluster] = view
    return view


def _populate_compact_cluster_views(label_to_id, view_cache, *cluster_count_maps):
    for count_map in cluster_count_maps:
        for cluster in count_map:
            _compact_cluster_view(cluster, label_to_id, view_cache)


def _prepare_compact_cluster_views(*cluster_count_maps):
    label_to_id = {}
    view_cache = {}
    _populate_compact_cluster_views(label_to_id, view_cache, *cluster_count_maps)
    return label_to_id, view_cache


def cluster_comparison_work(left_counts, right_counts):
    label_to_id, view_cache = _prepare_compact_cluster_views(left_counts, right_counts)
    return ClusterComparisonWork(label_to_id=label_to_id, view_cache=view_cache)


def _jaccard_distance_from_views(left_view, right_view):
    union_seed = left_view.size + right_view.size
    if union_seed == 0:
        return 0

    if len(left_view.counts) <= len(right_view.counts):
        smaller = left_view.counts
        larger = right_view.counts
    else:
        smaller = right_view.counts
        larger = left_view.counts
    intersection = 0
    for label_id, count in smaller.items():
        other_count = larger.get(label_id, 0)
        if other_count:
            intersection += min(count, other_count)
    union = union_seed - intersection
    return 1 - (intersection / union) if union else 0


def _cached_compact_jaccard_distance(left_cluster, right_cluster, cache, view_cache):
    if cache is not None:
        key = (left_cluster, right_cluster)
        cached = cache.get(key)
        if cached is not None:
            return cached
        reverse_key = (right_cluster, left_cluster)
        cached = cache.get(reverse_key)
        if cached is not None:
            return cached

    distance = _jaccard_distance_from_views(
        view_cache[left_cluster],
        view_cache[right_cluster],
    )
    if cache is not None:
        cache[key] = distance
    return distance


def _cluster_count_matrix(clusters, view_cache, label_count):
    matrix = np.zeros((len(clusters), label_count), dtype=np.int64)
    sizes = np.zeros(len(clusters), dtype=np.int64)
    for row_index, cluster in enumerate(clusters):
        view = view_cache[cluster]
        sizes[row_index] = view.size
        for label_id, count in view.items:
            matrix[row_index, label_id] = count
    return matrix, sizes


def _should_use_dense_jaccard_kernel(left_counts, right_counts, label_count):
    pair_count = len(left_counts) * len(right_counts)
    return (
        pair_count >= _DENSE_PAIR_THRESHOLD
        and label_count <= _DENSE_LABEL_LIMIT
        and pair_count * max(label_count, 1) <= _DENSE_WORK_LIMIT
    )


def _weighted_jaccard_distance_sum_sparse(left_counts, right_counts, jaccard_cache, view_cache):
    total = 0.0
    for left_cluster, left_multiplicity in left_counts.items():
        for right_cluster, right_multiplicity in right_counts.items():
            total += (
                left_multiplicity
                * right_multiplicity
                * _cached_compact_jaccard_distance(
                    left_cluster,
                    right_cluster,
                    jaccard_cache,
                    view_cache,
                )
            )
    return total


def _weighted_jaccard_distance_sum_dense(left_counts, right_counts, view_cache, label_count):
    left_clusters = tuple(left_counts)
    right_clusters = tuple(right_counts)
    left_matrix, left_sizes = _cluster_count_matrix(left_clusters, view_cache, label_count)
    right_matrix, right_sizes = _cluster_count_matrix(right_clusters, view_cache, label_count)
    right_multiplicities = np.fromiter(
        (right_counts[cluster] for cluster in right_clusters),
        dtype=float,
        count=len(right_clusters),
    )

    total = 0.0
    for left_index, left_cluster in enumerate(left_clusters):
        if label_count:
            intersections = np.minimum(left_matrix[left_index], right_matrix).sum(axis=1)
        else:
            intersections = np.zeros(len(right_clusters), dtype=np.int64)
        unions = left_sizes[left_index] + right_sizes - intersections
        with np.errstate(divide="ignore", invalid="ignore"):
            distances = np.where(unions > 0, 1 - (intersections / unions), 0.0)
        left_multiplicity = left_counts[left_cluster]
        total += float(left_multiplicity) * float(distances @ right_multiplicities)
    return total

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


def grf_comparison_metadata(left_counts, right_counts, left_cluster_set=None, right_cluster_set=None):
    if left_cluster_set is None:
        left_cluster_set = frozenset(left_counts)
    if right_cluster_set is None:
        right_cluster_set = frozenset(right_counts)
    return GrfComparisonMetadata(
        left_size=sum(left_counts.values()),
        right_size=sum(right_counts.values()),
        multiset_union_size=_multiset_union_size(left_counts, right_counts),
        left_minus_right=_multiset_difference_counts(left_counts, right_counts),
        right_minus_left=_multiset_difference_counts(right_counts, left_counts),
        set_union_size=len(left_cluster_set | right_cluster_set),
        left_only_counts=Counter(
            {
                cluster: count
                for cluster, count in left_counts.items()
                if cluster not in right_cluster_set
            }
        ),
        right_only_counts=Counter(
            {
                cluster: count
                for cluster, count in right_counts.items()
                if cluster not in left_cluster_set
            }
        ),
    )


def weighted_jaccard_distance_sum(left_counts, right_counts, jaccard_cache=None, *, work=None):
    if not left_counts or not right_counts:
        return 0.0

    if work is None:
        label_to_id, view_cache = _prepare_compact_cluster_views(left_counts, right_counts)
    else:
        label_to_id = work.label_to_id
        view_cache = work.view_cache
        _populate_compact_cluster_views(label_to_id, view_cache, left_counts, right_counts)
    if _should_use_dense_jaccard_kernel(left_counts, right_counts, len(label_to_id)):
        return _weighted_jaccard_distance_sum_dense(
            left_counts,
            right_counts,
            view_cache,
            len(label_to_id),
        )
    return _weighted_jaccard_distance_sum_sparse(
        left_counts,
        right_counts,
        jaccard_cache,
        view_cache,
    )


def _weighted_jaccard_distance_sum(left_counts, right_counts, jaccard_cache=None):
    return weighted_jaccard_distance_sum(
        left_counts,
        right_counts,
        jaccard_cache=jaccard_cache,
    )


def ext_grf_from_cluster_counts(A_counts, B_counts, jaccard_cache=None, *, work=None, metadata=None):
    if metadata is None:
        metadata = grf_comparison_metadata(A_counts, B_counts)

    if metadata.multiset_union_size == 0:
        return 0.0
    if metadata.left_size == 0 or metadata.right_size == 0:
        return 1.0

    num1 = weighted_jaccard_distance_sum(
        A_counts,
        metadata.right_minus_left,
        jaccard_cache,
        work=work,
    )
    num2 = weighted_jaccard_distance_sum(
        metadata.left_minus_right,
        B_counts,
        jaccard_cache,
        work=work,
    )

    return (num1 / (metadata.left_size * metadata.multiset_union_size)) + (
        num2 / (metadata.right_size * metadata.multiset_union_size)
    )


def legacy_set_grf_distance_from_cluster_contexts(
    true_cluster_context,
    reconstructed_cluster_context,
    *,
    jaccard_cache=None,
    work=None,
    metadata=None,
):
    if metadata is None:
        metadata = grf_comparison_metadata(
            true_cluster_context.counts,
            reconstructed_cluster_context.counts,
            true_cluster_context.cluster_set,
            reconstructed_cluster_context.cluster_set,
        )
    if metadata.set_union_size == 0:
        return 0.0
    if metadata.left_size == 0 or metadata.right_size == 0:
        return 1.0

    numerator_1 = weighted_jaccard_distance_sum(
        true_cluster_context.counts,
        metadata.right_only_counts,
        jaccard_cache=jaccard_cache,
        work=work,
    )
    numerator_2 = weighted_jaccard_distance_sum(
        reconstructed_cluster_context.counts,
        metadata.left_only_counts,
        jaccard_cache=jaccard_cache,
        work=work,
    )
    return (numerator_1 / (metadata.left_size * metadata.set_union_size)) + (
        numerator_2 / (metadata.right_size * metadata.set_union_size)
    )


def legacy_set_grf_similarity_from_cluster_contexts(
    true_cluster_context,
    reconstructed_cluster_context,
    *,
    jaccard_cache=None,
    work=None,
    metadata=None,
):
    return 1 - legacy_set_grf_distance_from_cluster_contexts(
        true_cluster_context,
        reconstructed_cluster_context,
        jaccard_cache=jaccard_cache,
        work=work,
        metadata=metadata,
    )


def exact_and_legacy_grf_from_cluster_contexts(
    true_cluster_context,
    reconstructed_cluster_context,
    *,
    jaccard_cache=None,
):
    work = cluster_comparison_work(
        true_cluster_context.counts,
        reconstructed_cluster_context.counts,
    )
    metadata = grf_comparison_metadata(
        true_cluster_context.counts,
        reconstructed_cluster_context.counts,
        true_cluster_context.cluster_set,
        reconstructed_cluster_context.cluster_set,
    )
    ext_grf_value = ext_grf_from_cluster_counts(
        true_cluster_context.counts,
        reconstructed_cluster_context.counts,
        jaccard_cache=jaccard_cache,
        work=work,
        metadata=metadata,
    )
    legacy_similarity = legacy_set_grf_similarity_from_cluster_contexts(
        true_cluster_context,
        reconstructed_cluster_context,
        jaccard_cache=jaccard_cache,
        work=work,
        metadata=metadata,
    )
    return ext_grf_value, legacy_similarity


def ext_grf_from_clusters(A, B, jaccard_cache=None):
    return ext_grf_from_cluster_counts(
        Counter(A),
        Counter(B),
        jaccard_cache=jaccard_cache,
    )


def _ext_grf_from_clusters(A, B):
    return ext_grf_from_clusters(A, B)


def ext_grf_tree(G1, root1, G2, root2):
    """
    Return the exact generalized Robinson-Foulds distance.

    Directionality: lower is better. The implementation follows the literature
    formula over multisets of clusters, so repeated equal clusters contribute
    with their multiplicity instead of being collapsed to set membership.
    """
    A = cluster_evaluation_context(G1, root1)
    B = cluster_evaluation_context(G2, root2)
    return ext_grf_from_cluster_counts(A.counts, B.counts)


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
