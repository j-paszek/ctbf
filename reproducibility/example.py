"""Tiny headless supplied-distance example. No external program or file output."""
from __future__ import annotations


def main() -> int:
    import networkx as nx
    import numpy as np
    from reconstructor_nj import neighbor_joining_classical
    from simulator import Genotype

    cells = [Genotype([2, i], node_id=i, cell_id=label)
             for i, label in enumerate(("A", "B", "C", "D"))]
    distances = np.array([[0, 5, 9, 9], [5, 0, 10, 10],
                          [9, 10, 0, 8], [9, 10, 8, 0]], dtype=float)
    tree, _, root = neighbor_joining_classical(distances, cells, max_id=3)
    if not nx.is_arborescence(tree) or tree.in_degree(root) != 0:
        raise RuntimeError("Reconstruction violated the rooted-tree contract")
    print(f"complete: toy reconstruction; {tree.number_of_nodes()} nodes; "
          f"{tree.number_of_edges()} edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
