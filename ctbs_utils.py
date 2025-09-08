import networkx as nx

def to_newick(tree: nx.DiGraph, prefer_weight: bool = True, use_events: bool = True, sort_children: bool = False) -> str:
    """
    Convert a directed tree (nx.DiGraph) into Newick format (iterative, fast).
    Node labels: tree.nodes[n]['cell_id'] (internal + leaves). If missing, 'None'.
    Branch lengths priority:
      1) 'weight' attribute (if prefer_weight and present),
      2) number of 'events' (if use_events and present),
      3) omitted if neither exists.

    Args
    ----
    tree : nx.DiGraph
        Rooted directed tree (each node has at most one parent).
    prefer_weight : bool
        If True, use 'weight' when available (reconstructor trees).
    use_events : bool
        If True, fall back to counting events from 'events' string (simulator trees).
    sort_children : bool
        If True, children are sorted by cell_id for deterministic output.

    Returns
    -------
    str
        Newick string ending with ';'
    """

    # --- Validate & find root (fast single scan) ---
    roots = [n for n, indeg in tree.in_degree() if indeg == 0]
    if len(roots) != 1:
        raise ValueError(f"Tree must have exactly one root (found {len(roots)})")
    root = roots[0]

    # --- Precompute branch length strings for all edges ---
    bl = {}
    for u, v, data in tree.edges(data=True):
        s = ""
        if prefer_weight and ("weight" in data) and (data["weight"] is not None):
            try:
                s = f":{float(data['weight']):.4f}"
            except Exception:
                s = ""
        elif use_events and isinstance(data.get("events"), str) and data["events"].strip():
            # Count non-empty event tokens
            cnt = sum(1 for tok in data["events"].split(";") if tok.strip())
            s = f":{cnt}"
        bl[(u, v)] = s

    # --- Child getter (optional sort for determinism) ---
    def get_children(n):
        ch = list(tree.successors(n))
        if sort_children:
            def keyf(c):
                cid = tree.nodes[c].get("cell_id")
                # Numeric sort if possible, else string
                try:
                    return (0, int(cid))
                except (TypeError, ValueError):
                    return (1, str(cid))
            ch.sort(key=keyf)
        return ch

    # --- Iterative DFS with explicit stack ---
    # Stack frames: (node, parent, stage, children, next_idx)
    # stage 0 = enter node; stage 1 = between/after children
    tokens = []
    stack = [(root, None, 0, None, 0)]

    while stack:
        node, parent, stage, children, idx = stack.pop()

        if stage == 0:
            label = str(tree.nodes[node].get("cell_id", "None"))
            children = get_children(node)
            if not children:
                # Leaf: label then branch length to parent (if any)
                tokens.append(label)
                if parent is not None:
                    tokens.append(bl.get((parent, node), ""))
            else:
                # Internal prelude: open paren, then process children one by one
                tokens.append("(")
                # We'll handle commas & close in stage 1
                stack.append((node, parent, 1, children, 0))
                # Push first child to process
                child = children[0]
                stack.append((child, node, 0, None, 0))

        else:  # stage == 1
            label = str(tree.nodes[node].get("cell_id", "None"))
            if idx < len(children) - 1:
                # More children to process: add comma, schedule next child
                tokens.append(",")
                stack.append((node, parent, 1, children, idx + 1))
                next_child = children[idx + 1]
                stack.append((next_child, node, 0, None, 0))
            else:
                # All children done: close, add node label, then branch length to parent
                tokens.append(")")
                tokens.append(label)
                if parent is not None:
                    tokens.append(bl.get((parent, node), ""))

    return "".join(tokens) + ";"
