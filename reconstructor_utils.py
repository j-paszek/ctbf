from collections import defaultdict

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from distance_semantics import parse_labeled_distance_matrix


def parse_distance_matrix(path):
    """Compatibility name for the shared strict distance-file parser."""
    return parse_labeled_distance_matrix(path)


def _cell_id_sort_key(tree, node_id):
    cell_id = tree.nodes[node_id].get("cell_id")
    return (cell_id is None, cell_id)


def visualize_tree_plotly(tree, node_levels=None, output_file="reconstructed.html", level_node_ordering=None):
    pos = {}
    level_to_nodes = defaultdict(list)

    # Group nodes by level and sort them
    for node, level in node_levels.items():
        node_id = node.node_id if hasattr(node, "node_id") else node
        level_to_nodes[level].append(node_id)

    for level in level_to_nodes:
        node_ids_in_level = level_to_nodes[level]
        if level_node_ordering and level in level_node_ordering:
            # Map from cell_id to node
            cell_id_to_node = {
                tree.nodes[node_id].get("cell_id"): node_id
                for node_id in node_ids_in_level
            }
            specified_ids = level_node_ordering[level]
            specified_nodes = [cell_id_to_node[cid] for cid in specified_ids if cid in cell_id_to_node]

            # Get remaining nodes not specified
            remaining_nodes = [
                node_id
                for node_id in node_ids_in_level
                if tree.nodes[node_id].get("cell_id") not in specified_ids
            ]
            remaining_nodes.sort(key=lambda node_id: _cell_id_sort_key(tree, node_id))  # optional sort of unspecified nodes

            # Combine specified + remaining
            level_to_nodes[level] = specified_nodes + remaining_nodes
        else:
            # Default: sort by cell_id
            level_to_nodes[level].sort(key=lambda node_id: _cell_id_sort_key(tree, node_id))

    # Assign x/y positions
    offset = 0.25
    max_level = len(level_to_nodes)
    z = 1
    for level, nodes in level_to_nodes.items():
        for i, node_id in enumerate(nodes):
            genome = tree.nodes[node_id].get("genome", np.array([None]))
            if genome.size == 1 and genome.flatten()[0] is None:
                if z % 2:
                    pos[node_id] = (offset, level)
                else:
                    pos[node_id] = (max_level - offset, level)
                offset += 0.5
                z += 1
            else:
                pos[node_id] = (i, level)

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
    fig.write_image(output_file + ".svg", width=1200, height=800, scale=2)
    fig.show()
