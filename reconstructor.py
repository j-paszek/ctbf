import itertools
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from simulator import Genotype


def parse_distance_matrix(path):
    with open(path) as f:
        n = int(f.readline())
        ids = []
        matrix = []
        for _ in range(n):
            parts = f.readline().strip().split()
            ids.append(int(parts[0]))
            matrix.append([float(x) for x in parts[1:]])
    return ids, np.array(matrix)

def build_evolution_tree(cell_lists, dist_matrix_path, r=2, only_nj=False):
    ids, full_dist_matrix = parse_distance_matrix(dist_matrix_path)
    id_to_index = {cid: i for i, cid in enumerate(ids)}
    unique_node_counter = itertools.count(start=max(ids) + 1)

    for lst in cell_lists:
        for cell in lst:
            cell.node_id = cell.cell_id  # Retained for compatibility

    tree = nx.DiGraph()
    node_levels = defaultdict(lambda: None)
    for level, lst in enumerate(cell_lists[::-1]):
        for cell in lst:
            node_levels[cell] = level
            if cell.node_id in tree.nodes:
                if not only_nj: # for simple NJ we ignore cell copies from different biopsies
                    new_node_id = next(unique_node_counter)
                    cell.node_id = new_node_id
                    tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)
            else:
                tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)

    for i in reversed(range(1, len(cell_lists))):
        upper, bottom = cell_lists[i - 1], cell_lists[i]
        for y in bottom:
            y_idx = id_to_index[y.cell_id]
            x_ks = []
            for x in upper:
                x_idx = id_to_index[x.cell_id]
                if full_dist_matrix[y_idx, x_idx] <= r:
                    x_ks.append(x)

            same_id_match = [x for x in x_ks if x.cell_id == y.cell_id]
            if same_id_match:
                x = same_id_match[0]
                tree.add_edge(x.node_id, y.node_id, weight=full_dist_matrix[y_idx, id_to_index[x.cell_id]])
                continue

            x_ks = [x for x in x_ks if not np.any((x.genome == 0) & (y.genome > 0))]

            if len(x_ks) == 1:
                x = x_ks[0]
                tree.add_edge(x.node_id, y.node_id, weight=full_dist_matrix[y_idx, id_to_index[x.cell_id]])
                continue

            if len(x_ks) > 1:
                closest = min(x_ks, key=lambda x: full_dist_matrix[y_idx, id_to_index[x.cell_id]])
                tree.add_edge(closest.node_id, y.node_id, weight=full_dist_matrix[y_idx, id_to_index[closest.cell_id]])
                continue

            # the case when x_ks empty there is no neighbour near
            new_node_id = next(unique_node_counter)
            copied_cell = Genotype(list(y.genome), y.cell_id)
            copied_cell.node_id = new_node_id
            cell_lists[i - 1].append(copied_cell)
            node_levels[copied_cell] = len(cell_lists) - i
            tree.add_node(copied_cell.node_id, genome=copied_cell.genome, cell_id=copied_cell.cell_id)
            tree.add_edge(copied_cell.node_id, y.node_id, weight=0)

    final_cells = cell_lists[0]
    final_ids = [cell.cell_id for cell in final_cells]
    dist_matrix = np.zeros((len(final_ids), len(final_ids)))
    for i, cell1 in enumerate(final_cells):
        for j, cell2 in enumerate(final_cells):
            idx1, idx2 = id_to_index[cell1.cell_id], id_to_index[cell2.cell_id]
            dist_matrix[i, j] = full_dist_matrix[idx1, idx2]

    max_id = next(unique_node_counter)
    tree, new_nodes = neighbor_joining(dist_matrix, final_cells, max_id, existing_tree=tree)

    for node in new_nodes:
        node_levels[node] = max(node_levels.values()) + 1

    return tree, node_levels


def neighbor_joining(dist_matrix, cells, max_id, existing_tree=None):
    D = dist_matrix.copy()
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}  # Store new nodes for visualization
    id_map = {i: cells[i] for i in range(len(cells))}
    next_id = max_id + 1

    for cell in cells:
        tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)

    while len(D) > 2:
        n = len(D)
        total_dist = D.sum(axis=1)
        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    Q[i][j] = (n - 2) * D[i][j] - total_dist[i] - total_dist[j]

        i, j = divmod(np.argmin(Q), n)
        if j < i:
            i, j = j, i

        delta = (total_dist[i] - total_dist[j]) / (n - 2)
        limb_len_i = 0.5 * (D[i][j] + delta)
        limb_len_j = 0.5 * (D[i][j] - delta)

        id_i, id_j = id_map[i], id_map[j]
        new_cell = Genotype(None, next_id)
        next_id += 1

        tree.add_node(new_cell.node_id, genome=new_cell.genome, cell_id=None)
        tree.add_edge(new_cell.node_id, id_i.node_id, weight=limb_len_i)
        tree.add_edge(new_cell.node_id, id_j.node_id, weight=limb_len_j)
        new_nodes[new_cell] = (id_i, id_j)

        new_row = [(D[i][k] + D[j][k] - D[i][j]) / 2 for k in range(n) if k != i and k != j]
        D = np.delete(D, [i, j], axis=0)
        D = np.delete(D, [i, j], axis=1)
        D = np.vstack([D, new_row])
        new_col = np.append(new_row, [0])[:, None]
        D = np.hstack([D, new_col])

        keys = [id_map[k] for k in range(n) if k != i and k != j]
        id_map = {k: v for k, v in enumerate(keys)}
        id_map[len(id_map)] = new_cell

    id1, id2 = id_map[0], id_map[1]
    root_cell = Genotype(None, -1)
    tree.add_node(root_cell.node_id, genome=root_cell.genome, cell_id=None)
    tree.add_edge(root_cell.node_id, id1.node_id, weight=D[0][1] / 2)
    tree.add_edge(root_cell.node_id, id2.node_id, weight=D[0][1] / 2)
    new_nodes[root_cell] = (id1, id2)

    return tree, new_nodes


def visualize_tree_plotly(tree, node_levels=None, output_file="reconstructed.html", level_node_ordering=None):
    pos = {}
    level_to_nodes = defaultdict(list)

    # Group nodes by level and sort them
    for node, level in node_levels.items():
        level_to_nodes[level].append(node)

    for level in level_to_nodes:
        nodes_in_level = level_to_nodes[level]
        if level_node_ordering and level in level_node_ordering:
            # Map from cell_id to node
            cell_id_to_node = {n.cell_id: n for n in nodes_in_level}
            specified_ids = level_node_ordering[level]
            specified_nodes = [cell_id_to_node[cid] for cid in specified_ids if cid in cell_id_to_node]

            # Get remaining nodes not specified
            remaining_nodes = [n for n in nodes_in_level if n.cell_id not in specified_ids]
            remaining_nodes.sort(key=lambda n: n.cell_id)  # optional sort of unspecified nodes

            # Combine specified + remaining
            level_to_nodes[level] = specified_nodes + remaining_nodes
        else:
            # Default: sort by cell_id
            level_to_nodes[level].sort(key=lambda n: n.cell_id)

    # Assign x/y positions
    offset = 0.25
    max_level = len(level_to_nodes)
    z = 1
    for level, nodes in level_to_nodes.items():
        for i, node in enumerate(nodes):
            if node.genome.size == 1 and node.genome.flatten()[0] is None:
                if z % 2:
                    pos[node.node_id] = (offset, level)
                else:
                    pos[node.node_id] = (max_level - offset, level)
                offset += 0.5
                z += 1
            else:
                pos[node.node_id] = (i, level)

    edge_x = []
    edge_y = []
    edge_label_pos_x, edge_label_pos_y = [], []
    edge_hover_labels = []
    edge_labels = []
    edge_marker_colors = []
    for (u, v), w in nx.get_edge_attributes(tree, 'weight').items():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])  # Add None to break the line
        edge_y.extend([y0, y1, None])
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        edge_label_pos_x.append(mid_x)
        edge_label_pos_y.append(mid_y)
        edge_labels.append("")  # Hide label by default
        # hover_edge_labels.append(str(event))  # Show label only on hover
        edge_marker_colors.append("green")
        edge_hover_labels.append(f"Distance: {w:.2f}")

        # Add markers for edge labels
    edge_l = go.Scatter(
            x=edge_label_pos_x, y=edge_label_pos_y, mode='markers+text',
            marker=dict(size=8, color=edge_marker_colors, opacity=0.5),  # Change color based on label presence
            text=edge_labels,
            hovertext=edge_hover_labels,  # Show edge label on hover
            textposition='middle center',
            hoverinfo='text'
        )

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    text = []

    for node, data in tree.nodes(data=True):
        gen_str = data.get("genome", "N/A")
        if gen_str.size == 1 and gen_str.flatten()[0] is None:
            gen_str = "N/A"
        cell_id = data.get("cell_id", "N/A")
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        label = f"cell_id={cell_id}<br>CN={gen_str}"
        text.append(label)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='lightblue',
            size=35,
            line_width=4),
        text=[data.get("cell_id", "N/A") for node, data in tree.nodes(data=True)],
        hovertext=text,
        textfont=dict(size=24)
    )

    pic=[]
    if level_node_ordering is not None:
        pic = [edge_trace, node_trace]
    else:
        pic = [edge_trace, node_trace, edge_l]

    fig = go.Figure(data=pic,
                   layout=go.Layout(
                       title=dict(
                           text='Reconstructed Tree',
                           font=dict(size=16)
                       ),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white',
                        paper_bgcolor='white')
                   )
    fig.write_html(output_file)
    fig.show()


if __name__ == '__main__':
    # CNPs here do not influence distances only for checking if descendant has x>0 where ancestor has x=0
    cell_lists = [
        [Genotype([2, 0, 1], 1), Genotype([1, 1, 1], 2)],
        [Genotype([2, 1, 1], 3), Genotype([1, 2, 0], 4)]
    ]
    cell_lists1 = [
        [Genotype([2, 2, 1], 1), Genotype([1, 1, 1], 2)],
        [Genotype([2, 1, 1], 3), Genotype([1, 2, 0], 4)]
    ]

    # 3->2
    # tree, a, b = build_evolution_tree(cell_lists, "distance_matrix.txt", r=2)
    # visualize_tree_plotly(tree, a, b)
    # 3->1
    # tree, a, b = build_evolution_tree(cell_lists1, "distance_matrix.txt", r=2)
    # visualize_tree_plotly(tree, a, b)
    # 3->2, 4->2
    tree, a = build_evolution_tree(cell_lists, "distance_matrix.txt", r=4)
    visualize_tree_plotly(tree, a)