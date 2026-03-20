import re
from collections import Counter
from typing import Tuple, Dict, Iterable, Optional, Any
import networkx as nx


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
    data = tree.nodes[node]
    label = data.get("cell_id")
    if label is None:
        return None
    s = str(label).strip()
    return s if s else None


def parent_of(tree: nx.DiGraph, node: Any) -> Optional[Any]:
    """Return parent node id or None. Assumes a rooted tree with single parent."""
    preds = list(tree.predecessors(node))
    if not preds:
        return None
    # If multiple parents (shouldn't happen), pick the first deterministically
    return preds[0]


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
    G = ensure_nx(tree_in)
    allowed = set(restrict_labels) if restrict_labels is not None else None

    # Collect named nodes
    nodes = [n for n in G.nodes if named_label(G, n) is not None]
    if allowed is not None:
        nodes = [n for n in nodes if named_label(G, n) in allowed]

    pairs = Counter()
    for desc in nodes:
        ld = named_label(G, desc)
        cur = parent_of(G, desc)
        while cur is not None:
            la = named_label(G, cur)
            if la is not None and (allowed is None or la in allowed):
                pairs[(la, ld)] += 1
            cur = parent_of(G, cur)
    return pairs


def label_edge_multiset(tree_in: Any,
                        restrict_labels: Optional[Iterable[str]] = None
                        ) -> Counter:
    """
    Count edges by (parent_label, child_label) for named parent & child nodes.
    """
    G = ensure_nx(tree_in)
    allowed = set(restrict_labels) if restrict_labels is not None else None
    edges = Counter()
    for node in G.nodes:
        # skip root nodes
        par = parent_of(G, node)
        if par is None:
            continue
        la = named_label(G, par)
        lb = named_label(G, node)
        if la is None or lb is None:
            continue
        if allowed is not None and (la not in allowed or lb not in allowed):
            continue
        edges[(la, lb)] += 1
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
    T = {k for k, v in true_ctr.items() if v > 0}
    if restrict_labels is not None:
        T = {(x, y) for (x, y) in T if x in restrict_labels and y in restrict_labels}
    R = {k for k, v in rec_ctr.items() if v > 0}
    tp = len(T & R)
    fp = len(R - T)
    fn = len(T - R)
    return tp, fp, fn, T, R


def prf1_iou(tp: int, fp: int, fn: int):
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
    Tt = ensure_nx(true_tree)
    Tr = ensure_nx(rec_tree)

    P_true = label_multiset_ancestor_pairs(Tt, restrict_labels)
    P_rec = label_multiset_ancestor_pairs(Tr, restrict_labels)

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
    original = evaluate_multiset(true_tree, rec_tree, restrict_labels=None)

    Tr = ensure_nx(rec_tree)
    if observed_labels is not None:
        V = set(observed_labels)
    else:
        V = {named_label(Tr, n) for n in Tr.nodes if named_label(Tr, n) is not None}

    pruned = evaluate_multiset(true_tree, rec_tree, restrict_labels=V)
    pruned["labels_used"] = sorted(V)
    return {"original": original, "pruned_truth": pruned}


def label_edge_multiset_wrapper(tree: Any,
                                restrict_labels: Optional[Iterable[str]] = None):
    """
    Backwards-compatible name for label_edge_multiset
    """
    return label_edge_multiset(tree, restrict_labels)


def evaluate_4(true_tree: Any,
               rec_tree: Any,
               restrict_labels: Optional[Iterable[str]] = None,
               print_debug: bool = False):
    """
    Full 4-mode evaluation. Inputs may be nx.DiGraph or Newick strings.
    Returns dict with keys 'ancestors_multiset','ancestors_unique','edges_multiset','edges_unique'.
    Each value is a dict with TP,FP,FN,precision,recall,F1,IoU and counts.
    """
    Tt = ensure_nx(true_tree)
    Tr = ensure_nx(rec_tree)

    # Ancestors multiset
    P_true_pairs = label_multiset_ancestor_pairs(Tt)
    P_rec_pairs = label_multiset_ancestor_pairs(Tr)
    tp1, fp1, fn1 = multiset_confusion_simple(P_true_pairs, P_rec_pairs)
    prec1, rec1, f11, iou1 = prf1_iou(tp1, fp1, fn1)
    mode1 = {
        "TP": tp1, "FP": fp1, "FN": fn1,
        "precision": prec1, "recall": rec1, "F1": f11, "IoU": iou1,
        "num_pairs_true": sum(P_true_pairs.values()),
        "num_pairs_rec": sum(P_rec_pairs.values())
    }

    # Ancestors unique (set)
    tp2, fp2, fn2, T2, R2 = _set_confusion(P_true_pairs, P_rec_pairs)
    prec2, rec2, f12, iou2 = prf1_iou(tp2, fp2, fn2)
    mode2 = {
        "TP": tp2, "FP": fp2, "FN": fn2,
        "precision": prec2, "recall": rec2, "F1": f12, "IoU": iou2,
        "num_unique_pairs_true": len(T2),
        "num_unique_pairs_rec": len(R2)
    }

    # Ancestors unique restricted (set)
    tp0, fp0, fn0, T0, R0 = _set_confusion(P_true_pairs, P_rec_pairs, restrict_labels)
    prec0, rec0, f10, iou0 = prf1_iou(tp0, fp0, fn0)
    mode0 = {
        "TP": tp0, "FP": fp0, "FN": fn0,
        "precision": prec0, "recall": rec0, "F1": f10, "IoU": iou0,
        "num_unique_pairs_true": len(T0),
        "num_unique_pairs_rec": len(R0)
    }

    # Edges multiset
    E_true = label_edge_multiset(Tt, restrict_labels)
    E_rec = label_edge_multiset(Tr, restrict_labels)
    tp3, fp3, fn3 = multiset_confusion_simple(E_true, E_rec)
    prec3, rec3, f13, iou3 = prf1_iou(tp3, fp3, fn3)
    mode3 = {
        "TP": tp3, "FP": fp3, "FN": fn3,
        "precision": prec3, "recall": rec3, "F1": f13, "IoU": iou3,
        "num_edges_true": sum(E_true.values()),
        "num_edges_rec": sum(E_rec.values())
    }

    # Edges unique (set)
    tp4, fp4, fn4, T4, R4 = _set_confusion(E_true, E_rec)
    prec4, rec4, f14, iou4 = prf1_iou(tp4, fp4, fn4)
    mode4 = {
        "TP": tp4, "FP": fp4, "FN": fn4,
        "precision": prec4, "recall": rec4, "F1": f14, "IoU": iou4,
        "num_unique_edges_true": len(T4),
        "num_unique_edges_rec": len(R4)
    }

    if print_debug:
        print("---- DEBUG four modes ----")
        print("Ancestors multiset:      TP/FP/FN =", tp1, fp1, fn1)
        print("Ancestors unique:        TP/FP/FN =", tp2, fp2, fn2)
        print("Edges multiset:          TP/FP/FN =", tp3, fp3, fn3)
        print("Edges unique:            TP/FP/FN =", tp4, fp4, fn4)
        print("anc unique & restricted: TP/FP/FN =", tp0, fp0, fn0)

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
