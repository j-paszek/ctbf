import networkx as nx
import re
import numpy as np

from reconstructor import visualize_tree_plotly


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


class TreeNode:
    def __init__(self, cell_id, node_id, genome=None):
        self.cell_id = cell_id
        self.node_id = node_id
        self.genome = genome if genome is not None else np.array([])

    def __repr__(self):
        return f"TreeNode({self.cell_id})"


def from_newick(newick_str: str):
    """
    Parse Newick string into:
    - tree: nx.DiGraph with string node_ids, storing attributes 'cell_id' and 'genome'
    - node_objects: dict {node_id: TreeNode} with full info
    """
    newick_str = newick_str.strip().rstrip(";")
    token_re = re.compile(r"\(|\)|,|[^(),:;]+(?::[0-9.eE+-]+)?")
    tokens = token_re.findall(newick_str)

    tree = nx.DiGraph()
    stack = []  # children in current group: [(node_obj, branch_length)]
    pending_internal = None
    node_counter = 0
    node_objects = {}

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        if token == "(":
            stack.append([])
        elif token == ",":
            continue
        elif token == ")":
            children = stack.pop()
            pending_internal = children
        else:
            # label[:length]
            if ":" in token:
                label, length = token.split(":", 1)
                label = label.strip()
                length = float(length)
            else:
                label = token.strip()
                length = None

            node_id = f"n{node_counter}"
            node_counter += 1
            node_obj = TreeNode(label, node_id)
            node_objects[node_id] = node_obj

            # add node to tree with attributes
            tree.add_node(node_id, cell_id=node_obj.cell_id, genome=node_obj.genome)

            # connect pending children
            if pending_internal is not None:
                for child_obj, child_len in pending_internal:
                    tree.add_edge(node_id, child_obj.node_id)
                    if child_len is not None:
                        tree[node_id][child_obj.node_id]["weight"] = child_len
                pending_internal = None

            # add current node to stack if inside another group
            if stack:
                stack[-1].append((node_obj, length))

    return tree, node_objects


def compute_node_levels(tree: nx.DiGraph, node_objects: dict):
    """
    Compute {TreeNode: level} mapping.
    """
    roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected 1 root, found {len(roots)}")
    root_id = roots[0]
    root_obj = node_objects[root_id]

    node_levels = {root_obj: 0}
    queue = [(root_id, root_obj)]

    while queue:
        parent_id, parent_obj = queue.pop(0)
        parent_level = node_levels[parent_obj]
        for child_id in tree.successors(parent_id):
            child_obj = node_objects[child_id]
            node_levels[child_obj] = parent_level + 1
            queue.append((child_id, child_obj))

    return node_levels


def vizualize_from_newick(newick_str):
    tree, no = from_newick(newick_str)
    nl = compute_node_levels(tree, no)
    max_level = max(nl.values())
    nl = {node: max_level - lvl for node, lvl in nl.items()}
    visualize_tree_plotly(tree, nl)


tt = "((((((((6)6,(58:2,39)39:2)6,((26)26)26:1)6)6)6:1,(((((41)41:1)18)18:1)12:1,((((1)1)1,((63:1)29,(64:2,44,66:2,67:1)44:1)29:1)1,(((45)45:1)20,((31)31)31:1)20:1)1)1)1)1:1,(((((((32,71:1)32)32:2,((8)8)8)8)8)8:1,(((((49)49:1,22)22)22:1,(((74:2)15,(52)52:1)15)15)15:1)0)0,((((((76:1,53)53:1,(24)24)24,((55)55:1,(56,81:1)56:1)37:1)24:1)16:1)5)5:1)0)0"
rt = "((((67:2.0000,26:2.0000,71:4.0000)31:2.0000,(63:1.0000,39:1.0000,1:1.0000)29:1.0000,(41:1.0000)18:2.0000,(32:3.0000,49:1.0000)15:2.0000,20:1.0000)1:0.0000,6:1.0000)1:0.0000,((24:1.0000,76:3.0000,53:2.0000)37:2.0000)16:2.0000)1"
njt= "((32:0.0000,71:1.0000)32:0.0000,(((53:0.0000,76:1.0000)53:0.0000,37:2.0000)53:0.0000,((26:0.0000,(((((((((1:0.0000,6:1.0000)1:0.0000,29:1.0000)1:0.0000,20:1.0000)1:0.0000,(16:0.0000,24:1.0000)16:2.0000)1:0.0000,31:2.0000)1:0.0000,(18:0.0000,41:1.0000)18:2.0000)1:0.0000,(15:0.0000,49:1.0000)15:2.0000)1:0.0000,63:2.0000)1:0.0000,67:2.0000)1:2.0000)26:0.0000,39:2.0000)26:2.0000)53:3.0000)32"

# vizualize_from_newick(tt)
vizualize_from_newick(rt)
# vizualize_from_newick(njt)