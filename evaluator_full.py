import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Tuple, Dict, FrozenSet, Iterable, Optional, Any
import networkx as nx


EVALUATE_4_VALUE_DIRECTION = {
    "TP": "count",
    "FP": "count",
    "FN": "count",
    "precision": "similarity",
    "recall": "similarity",
    "F1": "similarity",
    "IoU": "similarity",
}

EVALUATE_4_HIGHER_IS_BETTER = {
    "precision": True,
    "recall": True,
    "F1": True,
    "IoU": True,
}

EVALUATE_4_MODE_SPECS = {
    "ancestors_multiset": {
        "comparison": "ancestor-descendant label pairs with multiplicity",
        "kind": "similarity",
        "higher_is_better": True,
    },
    "ancestors_unique": {
        "comparison": "unique ancestor-descendant label pairs",
        "kind": "similarity",
        "higher_is_better": True,
    },
    "edges_multiset": {
        "comparison": "direct parent-child label edges with multiplicity",
        "kind": "similarity",
        "higher_is_better": True,
    },
    "edges_unique": {
        "comparison": "unique direct parent-child label edges",
        "kind": "similarity",
        "higher_is_better": True,
    },
    "ancestors_unique_restricted": {
        "comparison": "unique ancestor-descendant label pairs restricted to observed labels",
        "kind": "similarity",
        "higher_is_better": True,
        "paper_name": "AD-F1 when reading the F1 value",
    },
}


@dataclass(frozen=True)
class TreeEvaluationContext:
    graph: Optional[nx.DiGraph]
    labels: Dict[Any, str]
    parents: Dict[Any, Optional[Any]]
    children: Dict[Any, Tuple[Any, ...]]
    named_nodes: Tuple[Any, ...]
    roots: Tuple[Any, ...]


@dataclass
class RestrictedAdf1Cache:
    label_to_id: Dict[str, int] = field(default_factory=dict)
    true_pair_ids_by_restricted_labels: Dict[Optional[Tuple[Any, ...]], set] = field(
        default_factory=dict
    )


# ---------------------------
# --- Newick -> NetworkX
# ---------------------------
def from_newick(newick: str) -> nx.DiGraph:
    """
    Parse a Newick string into a NetworkX DiGraph where edges are parent->child.
    Node names (labels) are stored as node attribute 'cell_id'.
    Internal unnamed nodes get generated ids 'internal_0', ...
    Branch lengths (':x') are stored as edge attribute 'weight' if present.
    This parser handles typical Newick with branch lengths and internal labels.
    """
    s = newick.strip()
    if s.endswith(";"):
        s = s[:-1]

    # token regex: parentheses, commas, or labels with optional :length
    token_re = re.compile(r'\(|\)|,|[^(),\s:]+(?::[0-9.eE+-]+)?')
    tokens = token_re.findall(s)

    G = nx.DiGraph()
    stack = []  # stack of lists collecting children in current group
    pending_children = None
    internal_count = 0
    node_counter = 0

    # Helper to create a node id and add node with cell_id label
    def _make_node(label: Optional[str]) -> str:
        nonlocal node_counter, internal_count
        if label is None or label == "":
            nid = f"internal_{internal_count}"
            internal_count += 1
            G.add_node(nid, cell_id=None)
            return nid
        # ensure unique node id for same label => append counter
        nid = f"n{node_counter}"
        node_counter += 1
        G.add_node(nid, cell_id=str(label))
        return nid

    # We'll represent children temporarily as tuples (node_id, branch_length_or_None)
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok == "(":
            stack.append([])
        elif tok == ",":
            continue
        elif tok == ")":
            # finish clade: children are last list
            if not stack:
                raise ValueError("Malformed Newick (extra ')')")
            pending_children = stack.pop()
        else:
            # token is label or label:length
            if ":" in tok:
                label_part, length_part = tok.split(":", 1)
                label_part = label_part.strip()
                try:
                    length_val = float(length_part)
                except Exception:
                    length_val = None
            else:
                label_part = tok
                length_val = None

            label_part = label_part if label_part != "" else None
            node_id = _make_node(label_part)

            # If there are pending children (we just closed a bracket), connect them to this node
            if pending_children is not None:
                # pending_children is list of (child_node_id, child_length)
                for child_nid, child_len in pending_children:
                    G.add_edge(node_id, child_nid)
                    if child_len is not None:
                        G[node_id][child_nid]["weight"] = child_len
                pending_children = None

            # If inside a bracket, append this node as child for parent to be created later
            if stack:
                stack[-1].append((node_id, length_val))

    # If stack still non-empty, malformed
    if stack:
        # leftover children not attached -> create a root node and attach them
        root_id = _make_node("Root")
        for group in stack:
            for child_nid, child_len in group:
                G.add_edge(root_id, child_nid)
                if child_len is not None:
                    G[root_id][child_nid]["weight"] = child_len
    else:
        # If we created no explicit root and there is exactly one node without parents, that's root.
        roots = [n for n in G.nodes if G.in_degree(n) == 0]
        if len(roots) == 0 and len(G.nodes) > 0:
            # create a synthetic root if needed
            root_id = _make_node("Root")
            for n in list(G.nodes):
                if n != root_id and G.in_degree(n) == 0:
                    G.add_edge(root_id, n)

    return G


# ---------------------------
# --- Utilities
# ---------------------------
def ensure_nx(tree_or_newick: Any) -> nx.DiGraph:
    """
    If input is an nx.DiGraph -> return it.
    If input is a string -> parse as Newick and return nx.DiGraph.
    """
    if isinstance(tree_or_newick, nx.DiGraph):
        return tree_or_newick
    if isinstance(tree_or_newick, str):
        return from_newick(tree_or_newick)
    raise TypeError("Input must be networkx.DiGraph or Newick string")


def named_label(tree: nx.DiGraph, node: Any) -> Optional[str]:
    """
    Return label (cell_id) for node in tree.
    Node may be node id (hashable).
    If label is missing/None/empty return None.
    """
    return normalize_cell_label(tree.nodes[node].get("cell_id"))


def normalize_cell_label(label: Any) -> Optional[str]:
    if label is None:
        return None
    label = str(label).strip()
    return label if label else None


def normalize_cell_labels(labels: Optional[Iterable[Any]]) -> Optional[FrozenSet[str]]:
    """Apply the node-label canonicalization rule to a label collection."""
    if labels is None:
        return None
    values = (labels,) if isinstance(labels, str) else labels
    return frozenset(
        normalized_label
        for value in values
        if (normalized_label := normalize_cell_label(value)) is not None
    )


def parent_of(tree: nx.DiGraph, node: Any) -> Optional[Any]:
    """Return parent node id or None. Assumes a rooted tree with single parent."""
    # If multiple parents (shouldn't happen), pick the first deterministically.
    return next(iter(tree.predecessors(node)), None)


def _node_link_edges(data):
    if "links" in data:
        return data.get("links", [])
    return data.get("edges", [])


def tree_evaluation_context_from_node_link(data: Dict[str, Any]) -> TreeEvaluationContext:
    labels = {}
    parents = {}
    children = {}
    named_nodes = []

    for node in data.get("nodes", []):
        node_id = node.get("id")
        parents[node_id] = None
        children[node_id] = []
        label = normalize_cell_label(node.get("cell_id"))
        if label is not None:
            labels[node_id] = label
            named_nodes.append(node_id)

    for edge in _node_link_edges(data):
        source = edge.get("source")
        target = edge.get("target")
        if source not in children:
            children[source] = []
            parents.setdefault(source, None)
        if target not in parents:
            parents[target] = None
            children.setdefault(target, [])
        children[source].append(target)
        if parents[target] is None:
            parents[target] = source

    roots = tuple(node for node, parent in parents.items() if parent is None)
    return TreeEvaluationContext(
        graph=None,
        labels=labels,
        parents=parents,
        children={node: tuple(values) for node, values in children.items()},
        named_nodes=tuple(named_nodes),
        roots=roots,
    )


def tree_evaluation_context(tree_in: Any) -> TreeEvaluationContext:
    if isinstance(tree_in, TreeEvaluationContext):
        return tree_in
    if isinstance(tree_in, dict) and "nodes" in tree_in:
        return tree_evaluation_context_from_node_link(tree_in)
    G = ensure_nx(tree_in)
    labels = {}
    named_nodes = []
    for node, data in G.nodes(data=True):
        label = normalize_cell_label(data.get("cell_id"))
        if label:
            labels[node] = label
            named_nodes.append(node)

    parents = {}
    children = {}
    for node in G.nodes:
        parents[node] = parent_of(G, node)
        children[node] = tuple(G.successors(node))

    roots = tuple(node for node, parent in parents.items() if parent is None)
    return TreeEvaluationContext(
        graph=G,
        labels=labels,
        parents=parents,
        children=children,
        named_nodes=tuple(named_nodes),
        roots=roots,
    )


def ensure_tree_evaluation_context(tree_or_context: Any) -> TreeEvaluationContext:
    if isinstance(tree_or_context, TreeEvaluationContext):
        return tree_or_context
    return tree_evaluation_context(tree_or_context)


# ---------------------------
# --- Ancestor / Edge label multisets
# ---------------------------
def label_multiset_ancestor_pairs(tree_in: Any,
                                  restrict_labels: Optional[Iterable[str]] = None
                                  ) -> Counter:
    """
    For each named node (descendant) collect all named ancestors and count pairs (ancestor_label, descendant_label).
    Returns Counter of pairs -> multiplicity (multiset).
    Accepts either networkx.DiGraph or Newick string (will be converted).
    """
    context = ensure_tree_evaluation_context(tree_in)
    allowed = normalize_cell_labels(restrict_labels)

    if not context.roots:
        return _label_multiset_ancestor_pairs_by_parent_chain(context, allowed)

    pairs = Counter()
    ancestor_counts = Counter()
    for root in context.roots:
        stack = [(root, None, False)]
        while stack:
            node, added_label, leaving = stack.pop()
            if leaving:
                if added_label is not None:
                    ancestor_counts[added_label] -= 1
                    if ancestor_counts[added_label] == 0:
                        del ancestor_counts[added_label]
                continue

            label = context.labels.get(node)
            active_label = label if label is not None and (allowed is None or label in allowed) else None
            if active_label is not None:
                for ancestor_label, count in ancestor_counts.items():
                    pairs[(ancestor_label, active_label)] += count
                ancestor_counts[active_label] += 1

            stack.append((node, active_label, True))
            for child in reversed(context.children.get(node, ())):
                stack.append((child, None, False))
    return pairs


def _label_multiset_ancestor_pairs_by_parent_chain(context, allowed):
    nodes = context.named_nodes
    if allowed is not None:
        nodes = [node for node in nodes if context.labels[node] in allowed]

    pairs = Counter()
    for desc in nodes:
        descendant_label = context.labels[desc]
        cur = context.parents.get(desc)
        while cur is not None:
            ancestor_label = context.labels.get(cur)
            if ancestor_label is not None and (allowed is None or ancestor_label in allowed):
                pairs[(ancestor_label, descendant_label)] += 1
            cur = context.parents.get(cur)
    return pairs


def _intern_label(label, label_to_id):
    if label_to_id is None:
        return label
    existing = label_to_id.get(label)
    if existing is not None:
        return existing
    label_id = len(label_to_id)
    label_to_id[label] = label_id
    return label_id


def _ancestor_pair_id(ancestor_id, descendant_id):
    pair_sum = ancestor_id + descendant_id
    return (pair_sum * (pair_sum + 1) // 2) + descendant_id


def _restricted_labels_cache_key(restrict_labels: Optional[Iterable[str]]):
    allowed = normalize_cell_labels(restrict_labels)
    return None if allowed is None else tuple(sorted(allowed))


def unique_ancestor_pair_set(tree_in: Any,
                             restrict_labels: Optional[Iterable[str]] = None,
                             *,
                             label_to_id: Optional[Dict[str, int]] = None):
    context = ensure_tree_evaluation_context(tree_in)
    allowed = normalize_cell_labels(restrict_labels)

    if not context.roots:
        pairs = set()
        nodes = context.named_nodes
        if allowed is not None:
            nodes = [node for node in nodes if context.labels[node] in allowed]
        for desc in nodes:
            descendant_label = context.labels[desc]
            current = context.parents.get(desc)
            while current is not None:
                ancestor_label = context.labels.get(current)
                if ancestor_label is not None and (allowed is None or ancestor_label in allowed):
                    pairs.add((
                        _intern_label(ancestor_label, label_to_id),
                        _intern_label(descendant_label, label_to_id),
                    ))
                current = context.parents.get(current)
        return pairs

    pairs = set()
    ancestor_labels = Counter()
    for root in context.roots:
        stack = [(root, None, False)]
        while stack:
            node, added_label, leaving = stack.pop()
            if leaving:
                if added_label is not None:
                    ancestor_labels[added_label] -= 1
                    if ancestor_labels[added_label] == 0:
                        del ancestor_labels[added_label]
                continue

            label = context.labels.get(node)
            active_label = label if label is not None and (allowed is None or label in allowed) else None
            if active_label is not None:
                active_label_id = _intern_label(active_label, label_to_id)
                for ancestor_label in ancestor_labels:
                    pairs.add((
                        _intern_label(ancestor_label, label_to_id),
                        active_label_id,
                    ))
                ancestor_labels[active_label] += 1

            stack.append((node, active_label, True))
            for child in reversed(context.children.get(node, ())):
                stack.append((child, None, False))
    return pairs


def unique_ancestor_pair_id_set(tree_in: Any,
                                restrict_labels: Optional[Iterable[str]] = None,
                                *,
                                label_to_id: Optional[Dict[str, int]] = None):
    context = ensure_tree_evaluation_context(tree_in)
    allowed = normalize_cell_labels(restrict_labels)
    if label_to_id is None:
        label_to_id = {}

    if not context.roots:
        pairs = set()
        nodes = context.named_nodes
        if allowed is not None:
            nodes = [node for node in nodes if context.labels[node] in allowed]
        for desc in nodes:
            descendant_id = _intern_label(context.labels[desc], label_to_id)
            current = context.parents.get(desc)
            while current is not None:
                ancestor_label = context.labels.get(current)
                if ancestor_label is not None and (allowed is None or ancestor_label in allowed):
                    ancestor_id = _intern_label(ancestor_label, label_to_id)
                    pairs.add(_ancestor_pair_id(ancestor_id, descendant_id))
                current = context.parents.get(current)
        return pairs

    pairs = set()
    ancestor_label_ids = Counter()
    for root in context.roots:
        stack = [(root, None, False)]
        while stack:
            node, added_label_id, leaving = stack.pop()
            if leaving:
                if added_label_id is not None:
                    ancestor_label_ids[added_label_id] -= 1
                    if ancestor_label_ids[added_label_id] == 0:
                        del ancestor_label_ids[added_label_id]
                continue

            label = context.labels.get(node)
            active_label_id = None
            if label is not None and (allowed is None or label in allowed):
                active_label_id = _intern_label(label, label_to_id)
                for ancestor_label_id in ancestor_label_ids:
                    pairs.add(_ancestor_pair_id(ancestor_label_id, active_label_id))
                ancestor_label_ids[active_label_id] += 1

            stack.append((node, active_label_id, True))
            for child in reversed(context.children.get(node, ())):
                stack.append((child, None, False))
    return pairs


def label_edge_multiset(tree_in: Any,
                        restrict_labels: Optional[Iterable[str]] = None
                        ) -> Counter:
    """
    Count edges by (parent_label, child_label) for named parent & child nodes.
    """
    context = ensure_tree_evaluation_context(tree_in)
    allowed = normalize_cell_labels(restrict_labels)
    edges = Counter()
    for node, parent in context.parents.items():
        if parent is None:
            continue
        parent_label = context.labels.get(parent)
        child_label = context.labels.get(node)
        if parent_label is None or child_label is None:
            continue
        if allowed is not None and (parent_label not in allowed or child_label not in allowed):
            continue
        edges[(parent_label, child_label)] += 1
    return edges


# ---------------------------
# --- Confusion / metrics (unchanged semantics)
# ---------------------------
def multiset_confusion_simple(true_pairs: Counter, rec_pairs: Counter) -> Tuple[int, int, int]:
    """
    Simple multiset confusion (older variant in your code).
    Returns tp, fp, fn (counts over multiplicities).
    """
    all_keys = set(true_pairs) | set(rec_pairs)
    tp = fp = fn = 0
    for k in all_keys:
        ct = true_pairs.get(k, 0)
        cr = rec_pairs.get(k, 0)
        tp += min(ct, cr)
        if cr > ct:
            fp += (cr - ct)
        elif ct > cr:
            fn += (ct - cr)
    return tp, fp, fn


def multiset_confusion(true_pairs: Counter,
                       rec_pairs: Counter,
                       return_details: bool = False,
                       as_lists: bool = False):
    """
    Full multiset confusion. If return_details is False returns (tp,fp,fn).
    If return_details True returns (tp,fp,fn, tp_ctr, fp_ctr, fn_ctr)
    If as_lists True returns (tp,fp,fn, tp_list, fp_list, fn_list)
    """
    tp_count = fp_count = fn_count = 0
    tp_ctr = Counter()
    fp_ctr = Counter()
    fn_ctr = Counter()

    all_keys = set(true_pairs) | set(rec_pairs)
    for k in all_keys:
        ct = true_pairs.get(k, 0)
        cr = rec_pairs.get(k, 0)

        m = min(ct, cr)
        if m:
            tp_ctr[k] = m
            tp_count += m

        if cr > ct:
            d = cr - ct
            fp_ctr[k] = d
            fp_count += d

        if ct > cr:
            d = ct - cr
            fn_ctr[k] = d
            fn_count += d

    if not return_details:
        return tp_count, fp_count, fn_count

    if as_lists:
        def _counter_to_list(counter: Counter):
            out = []
            for pair, cnt in counter.items():
                out.extend([pair] * cnt)
            return out

        tp_list = _counter_to_list(tp_ctr)
        fp_list = _counter_to_list(fp_ctr)
        fn_list = _counter_to_list(fn_ctr)
        return tp_count, fp_count, fn_count, tp_list, fp_list, fn_list

    return tp_count, fp_count, fn_count, tp_ctr, fp_ctr, fn_ctr


def _set_confusion(true_ctr: Counter, rec_ctr: Counter, restrict_labels: Optional[Iterable[str]] = None):
    """
    Set-level confusion (unique pairs only).
    Returns tp, fp, fn, set_true, set_rec
    """
    return _set_confusion_from_sets(
        _positive_key_set(true_ctr),
        _positive_key_set(rec_ctr),
        restrict_labels=restrict_labels,
    )


def _positive_key_set(counter: Counter):
    return {key for key, value in counter.items() if value > 0}


def _set_confusion_from_sets(true_set, rec_set, restrict_labels: Optional[Iterable[str]] = None):
    T = true_set
    if restrict_labels is not None:
        allowed = normalize_cell_labels(restrict_labels)
        T = {(x, y) for (x, y) in true_set if x in allowed and y in allowed}
    R = rec_set
    tp = len(T & R)
    fp = len(R - T)
    fn = len(T - R)
    return tp, fp, fn, T, R


def prf1_iou(tp: int, fp: int, fn: int):
    """
    Return precision, recall, F1, and IoU similarity scores.

    Directionality: higher is better for all four values. TP/FP/FN remain
    raw counts and are not themselves similarity or distance metrics.
    """
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return prec, rec, f1, iou


# ---------------------------
# --- Evaluation wrappers (take nx.DiGraph or Newick string)
# ---------------------------
def evaluate_multiset(true_tree: Any, rec_tree: Any,
                      restrict_labels: Optional[Iterable[str]] = None):
    """
    Compute multiset ancestor-pair evaluation (returns dict like original).
    true_tree and rec_tree may be nx.DiGraph or Newick strings.
    """
    true_context = ensure_tree_evaluation_context(true_tree)
    rec_context = ensure_tree_evaluation_context(rec_tree)

    P_true = label_multiset_ancestor_pairs(true_context, restrict_labels)
    P_rec = label_multiset_ancestor_pairs(rec_context, restrict_labels)

    tp, fp, fn, tp_ctr, fp_ctr, fn_ctr = multiset_confusion(P_true, P_rec, return_details=True, as_lists=False)
    prec, rec, f1, iou = prf1_iou(tp, fp, fn)

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "precision": prec, "recall": rec, "F1": f1, "IoU": iou,
        "num_pairs_true": sum(P_true.values()),
        "num_pairs_rec": sum(P_rec.values())
    }


def evaluate_multiset_with_pruned_truth(true_tree: Any,
                                        rec_tree: Any,
                                        observed_labels: Optional[Iterable[str]] = None):
    """
    Run original evaluation and also evaluation restricted to observed labels from rec_tree (or provided set).
    """
    true_context = ensure_tree_evaluation_context(true_tree)
    rec_context = ensure_tree_evaluation_context(rec_tree)

    original = evaluate_multiset(true_context, rec_context, restrict_labels=None)
    if observed_labels is not None:
        V = normalize_cell_labels(observed_labels)
    else:
        V = normalize_cell_labels(rec_context.labels.values())

    pruned = evaluate_multiset(true_context, rec_context, restrict_labels=V)
    pruned["labels_used"] = sorted(V)
    return {"original": original, "pruned_truth": pruned}


def label_edge_multiset_wrapper(tree: Any,
                                restrict_labels: Optional[Iterable[str]] = None):
    """
    Backwards-compatible name for label_edge_multiset
    """
    return label_edge_multiset(tree, restrict_labels)


def _multiset_mode(true_counter, rec_counter, true_count_key, rec_count_key):
    tp, fp, fn = multiset_confusion_simple(true_counter, rec_counter)
    precision, recall, f1, iou = prf1_iou(tp, fp, fn)
    return {
        "TP": tp, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall, "F1": f1, "IoU": iou,
        true_count_key: sum(true_counter.values()),
        rec_count_key: sum(rec_counter.values()),
    }


def _unique_mode_from_sets(true_set, rec_set, true_count_key, rec_count_key, restrict_labels=None):
    tp, fp, fn, filtered_true_set, filtered_rec_set = _set_confusion_from_sets(
        true_set,
        rec_set,
        restrict_labels=restrict_labels,
    )
    precision, recall, f1, iou = prf1_iou(tp, fp, fn)
    return {
        "TP": tp, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall, "F1": f1, "IoU": iou,
        true_count_key: len(filtered_true_set),
        rec_count_key: len(filtered_rec_set),
    }, filtered_true_set, filtered_rec_set


def _intersection_size(left_set, right_set):
    if len(left_set) > len(right_set):
        left_set, right_set = right_set, left_set
    return sum(1 for item in left_set if item in right_set)


def _unique_mode_counts_from_sets(true_set, rec_set, true_count_key, rec_count_key):
    tp = _intersection_size(true_set, rec_set)
    fp = len(rec_set) - tp
    fn = len(true_set) - tp
    precision, recall, f1, iou = prf1_iou(tp, fp, fn)
    return {
        "TP": tp, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall, "F1": f1, "IoU": iou,
        true_count_key: len(true_set),
        rec_count_key: len(rec_set),
    }


def adf1_restricted_metrics_from_contexts(true_tree: Any,
                                          rec_tree: Any,
                                          restrict_labels: Optional[Iterable[str]] = None,
                                          *,
                                          cache: Optional[RestrictedAdf1Cache] = None):
    true_context = ensure_tree_evaluation_context(true_tree)
    rec_context = ensure_tree_evaluation_context(rec_tree)
    restricted_label_set = normalize_cell_labels(restrict_labels)
    if cache is None:
        label_to_id = {}
        true_pairs = unique_ancestor_pair_id_set(
            true_context,
            restrict_labels=restricted_label_set,
            label_to_id=label_to_id,
        )
    else:
        label_to_id = cache.label_to_id
        cache_key = _restricted_labels_cache_key(restricted_label_set)
        true_pairs = cache.true_pair_ids_by_restricted_labels.get(cache_key)
        if true_pairs is None:
            true_pairs = unique_ancestor_pair_id_set(
                true_context,
                restrict_labels=restricted_label_set,
                label_to_id=label_to_id,
            )
            cache.true_pair_ids_by_restricted_labels[cache_key] = true_pairs

    rec_pairs = unique_ancestor_pair_id_set(rec_context, label_to_id=label_to_id)
    return _unique_mode_counts_from_sets(
        true_pairs,
        rec_pairs,
        "num_unique_pairs_true",
        "num_unique_pairs_rec",
    )


def ancestors_unique_restricted_metrics(true_tree: Any,
                                        rec_tree: Any,
                                        restrict_labels: Optional[Iterable[str]] = None):
    return adf1_restricted_metrics_from_contexts(
        true_tree,
        rec_tree,
        restrict_labels=restrict_labels,
    )


def evaluate_4(true_tree: Any,
               rec_tree: Any,
               restrict_labels: Optional[Iterable[str]] = None,
               print_debug: bool = False):
    """
    Full 4-mode evaluation. Inputs may be nx.DiGraph or Newick strings.
    Returns dict with keys 'ancestors_multiset','ancestors_unique','edges_multiset',
    'edges_unique', and 'ancestors_unique_restricted'.
    Each value is a dict with TP,FP,FN,precision,recall,F1,IoU and counts.

    Directionality: precision, recall, F1, and IoU are similarity scores where
    higher is better. The paper-facing AD-F1 metric is
    result['ancestors_unique_restricted']['F1'].
    """
    true_context = ensure_tree_evaluation_context(true_tree)
    rec_context = ensure_tree_evaluation_context(rec_tree)

    # Ancestors multiset
    P_true_pairs = label_multiset_ancestor_pairs(true_context)
    P_rec_pairs = label_multiset_ancestor_pairs(rec_context)
    mode1 = _multiset_mode(P_true_pairs, P_rec_pairs, "num_pairs_true", "num_pairs_rec")

    true_pair_set = _positive_key_set(P_true_pairs)
    rec_pair_set = _positive_key_set(P_rec_pairs)

    # Ancestors unique (set)
    mode2, T2, R2 = _unique_mode_from_sets(
        true_pair_set,
        rec_pair_set,
        "num_unique_pairs_true",
        "num_unique_pairs_rec",
    )

    # Ancestors unique restricted (set)
    mode0, T0, R0 = _unique_mode_from_sets(
        true_pair_set,
        rec_pair_set,
        "num_unique_pairs_true",
        "num_unique_pairs_rec",
        restrict_labels=restrict_labels,
    )

    # Edges multiset
    E_true = label_edge_multiset(true_context, restrict_labels)
    E_rec = label_edge_multiset(rec_context, restrict_labels)
    mode3 = _multiset_mode(E_true, E_rec, "num_edges_true", "num_edges_rec")

    # Edges unique (set)
    mode4, T4, R4 = _unique_mode_from_sets(
        _positive_key_set(E_true),
        _positive_key_set(E_rec),
        "num_unique_edges_true",
        "num_unique_edges_rec",
    )

    if print_debug:
        print("---- DEBUG four modes ----")
        print("Ancestors multiset:      TP/FP/FN =", mode1["TP"], mode1["FP"], mode1["FN"])
        print("Ancestors unique:        TP/FP/FN =", mode2["TP"], mode2["FP"], mode2["FN"])
        print("Edges multiset:          TP/FP/FN =", mode3["TP"], mode3["FP"], mode3["FN"])
        print("Edges unique:            TP/FP/FN =", mode4["TP"], mode4["FP"], mode4["FN"])
        print("anc unique & restricted: TP/FP/FN =", mode0["TP"], mode0["FP"], mode0["FN"])

        _, _, _, tp_list, fp_list, fn_list = multiset_confusion(
            P_true_pairs, P_rec_pairs,
            return_details=True, as_lists=True
        )
        print("MODE 1")
        print(f"TP ({len(tp_list)}):{sorted(tp_list)}")
        print(f"FP ({len(fp_list)}):{sorted(fp_list)}")
        print(f"FN ({len(fn_list)}):{sorted(fn_list)}")

        print("MODE 2")
        tp_pairs_unique = T2 & R2
        fp_pairs_unique = R2 - T2
        fn_pairs_unique = T2 - R2
        print(f"TP ({len(tp_pairs_unique)}): {sorted(tp_pairs_unique)}")
        print(f"FP ({len(fp_pairs_unique)}): {sorted(fp_pairs_unique)}")
        print(f"FN ({len(fn_pairs_unique)}): {sorted(fn_pairs_unique)}")

        print("MODE 0")
        tp_pairs_unique_r = T0 & R0
        fp_pairs_unique_r = R0 - T0
        fn_pairs_unique_r = T0 - R0
        print(f"TP ({len(tp_pairs_unique_r)}): {sorted(tp_pairs_unique_r)}")
        print(f"FP ({len(fp_pairs_unique_r)}): {sorted(fp_pairs_unique_r)}")
        print(f"FN ({len(fn_pairs_unique_r)}): {sorted(fn_pairs_unique_r)}")

    return {
        "ancestors_multiset": mode1,
        "ancestors_unique": mode2,
        "edges_multiset": mode3,
        "edges_unique": mode4,
        "ancestors_unique_restricted": mode0
    }


# ---------------------------
# --- Printing helpers (left unchanged)
# ---------------------------
def print_table(results, file):
    rows = ["precision", "recall", "F1", "IoU", "TP", "FP", "FN"]
    ts = ["T", "Trec", "Tnj"]
    for result in results:
        for t in ts:
            file.write(t + "\t" + result["trees"][t] + "\n")
        res = result["results"]
        for row in rows:
            file.write(row + "\t" + str(round(res[0][row], 3)).replace(".", ",") + "\t" + str(res[1][row]).replace(".",
                                                                                                                   ",") + "\n")
        file.write("seed\t" + str(result["seed"]) + "\n")


def print_table2(results, file):
    cols = ["T", "Trec", "Tnj", "precision", "recall", "F1", "IoU", "TP", "FP", "FN", "precision_pruned",
            "recall_pruned", "F1_pruned", "IoU_pruned", "TP_pruned", "FP_pruned", "FN_pruned", "seed"]
    c = ["precision", "recall", "F1", "IoU", "TP", "FP", "FN"]
    file.write("\t".join(cols + ["\n"]))
    for result in results:
        res_rec, res_nj = result["results"]
        file.write("\t".join([result["trees"]["T"], result["trees"]["Trec"], "-", "\t".join(
            [str(round(res_rec[type][k], 3)).replace(".", ",") for type in ['original', 'pruned_truth'] for k in c]),
                              str(result["seed"]) + "\n"]))
        file.write("\t".join([result["trees"]["T"], "-", result["trees"]["Tnj"], "\t".join(
            [str(round(res_nj[type][k], 3)).replace(".", ",") for type in ['original', 'pruned_truth'] for k in c]),
                              str(result["seed"]) + "\n"]))


# ---------------------------
# --- Quick test (example)
# ---------------------------
if __name__ == "__main__":
    # Example Newick strings (similar to previous examples)
    newick_true = "((A:1,B:1)X:0.5,(C:1,D:1)Y:0.5)Root:0;"
    newick_rec = "((A,B),(C,D))Root;"

    # Parse into nx.DiGraph (or you would use your prebuilt nx trees)
    true_G = from_newick(newick_true)
    rec_G = from_newick(newick_rec)

    # Evaluate full 4 modes
    res = evaluate_4(true_G, rec_G, print_debug=True)
    print("\nRESULTS:")
    import pprint;

    pprint.pprint(res)
