"""Small browser gallery for the quick-start example; no bundled JavaScript library."""
from __future__ import annotations

from collections import defaultdict
from html import escape

import networkx as nx
import numpy as np
import plotly.graph_objects as go


STYLE = """
body {max-width:1200px; margin:2rem auto; padding:0 1rem; color:#263442;
      font:16px/1.55 system-ui,sans-serif; background:#fff;}
a {color:#245c85;} nav a {margin-right:1.2rem;} h1 {font-size:1.7rem;}
table {border-collapse:collapse; width:100%; margin:1.5rem 0;}
th,td {border-bottom:1px solid #d8e0e7; text-align:left; padding:.7rem;}
footer {font-size:.85rem; margin-top:2rem; color:#546371;}
"""


def positions(tree):
    root = next(n for n, d in tree.in_degree() if d == 0)
    depths = nx.single_source_shortest_path_length(tree, root)
    order = list(nx.dfs_preorder_nodes(tree, root))
    xs, leaf = {}, 0
    for node in reversed(order):
        children = list(tree.successors(node))
        if children:
            xs[node] = sum(xs[c] for c in children) / len(children)
        else:
            xs[node], leaf = leaf, leaf + 1
    return {node: (xs[node], -depths[node]) for node in order}


def figure(tree, title, biopsies, *, truth_view):
    positions_by_id = positions(tree)
    occurrences, state_biopsies, profiles = defaultdict(list), defaultdict(list), {}
    for index, biopsy in enumerate(biopsies, 1):
        for cell in biopsy:
            occurrences[cell.node_id].append(index)
            state_biopsies[str(cell.cell_id)].append(index)
            profiles[str(cell.cell_id)] = cell.genome
    edge_x, edge_y, hover_x, hover_y, hover = [], [], [], [], []
    for u, v, attrs in tree.edges(data=True):
        x0, y0 = positions_by_id[u]
        x1, y1 = positions_by_id[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        hover_x.append((x0+x1)/2)
        hover_y.append((y0+y1)/2)
        if "events_text" in attrs:
            text = str(attrs["events_text"]) or "No net change"
        elif "weight" in attrs:
            text = f"Edge weight: {attrs['weight']:.3g}"
        else:
            text = "Ancestry edge (may represent a contracted path)"
        hover.append(escape(text))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#a8b2bd", width=1.4),
                            hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=hover_x, y=hover_y, mode="markers", hovertext=hover,
                            hoverinfo="text", marker=dict(size=8, color="#adb9c4", opacity=.4),
                            showlegend=False))
    groups = defaultdict(list)
    for node, attrs in tree.nodes(data=True):
        label = attrs.get("cell_id")
        group = ("Unlabeled vertex" if label is None else
                 "Sampled occurrence" if truth_view and node in occurrences else
                 "Observed CNP state" if str(label) in state_biopsies else "Other true CNP state")
        groups[group].append(node)
    colors = {"Unlabeled vertex": "#ffffff", "Sampled occurrence": "#27765b",
              "Observed CNP state": "#c4dced", "Other true CNP state": "#e6e9ed"}
    for group, nodes in groups.items():
        texts, hovers = [], []
        for node in nodes:
            attrs = tree.nodes[node]
            label = attrs.get("cell_id")
            texts.append("–" if label is None else str(label))
            cn = attrs.get("genome")
            if cn is None:
                cn = profiles.get(str(label)) if label is not None else None
            details = [f"Vertex: {escape(str(node))}", f"CNP label: {escape(str(label)) if label is not None else 'unlabeled'}"]
            if label is not None and cn is not None:
                values = np.asarray(cn).reshape(-1).tolist()
                chunks = [", ".join(str(x) for x in values[i:i+20]) for i in range(0, len(values), 20)]
                details.append("Copy numbers:<br>" + "<br>".join(escape(c) for c in chunks))
            if truth_view and node in occurrences:
                details.append("Sampled in biopsy: " + ", ".join(map(str, occurrences[node])))
            if str(label) in state_biopsies:
                details.append("State observed in biopsy: " + ", ".join(map(str, state_biopsies[str(label)])))
            hovers.append("<br>".join(details))
        fig.add_trace(go.Scatter(
            x=[positions_by_id[n][0] for n in nodes], y=[positions_by_id[n][1] for n in nodes],
            mode="markers+text", text=texts, textposition="middle center", hovertext=hovers,
            hoverinfo="text", name=group, textfont=dict(size=11, color="#fff" if group == "Sampled occurrence" else "#243746"),
            marker=dict(size=25, color=colors[group], line=dict(color="#62788a", width=1)),
        ))
    fig.update_layout(title=title, height=800, template="plotly_white", hovermode="closest",
                      margin=dict(l=25, r=25, t=60, b=60),
                      legend=dict(orientation="h", y=-.06),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def page(title, body):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{escape(title)}</title><style>{STYLE}</style></head><body>{body}'
            '<footer>CTBF tutorial · Software: MIT · Original simulated data and CTBF trees: '
            '<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · '
            'CTBF contributors. Interactive plots load Plotly.js from its CDN.</footer></body></html>')


def write_gallery(output, trees, biopsies, methods, summary):
    titles = {"true_tree": "True tree · complete simulated history",
              "trimmed_true_tree": "Trimmed true tree · observed CNP states",
              **{key: f"{key} · {spec[0]}" for key, spec in methods.items() if key in trees}}
    nav = '<nav><a href="index.html">Overview</a>' + "".join(
        f'<a href="{key}.html">{escape(key.replace("_", " "))}</a>' for key in trees) + '</nav>'
    for key, tree in trees.items():
        chart = figure(tree, titles[key], biopsies, truth_view=key in {"true_tree", "trimmed_true_tree"})
        plot = chart.to_html(full_html=False, include_plotlyjs="cdn", div_id=f"ctbf-{key}",
                             config={"displaylogo": False, "responsive": True})
        body = nav + f'<h1>{escape(titles[key])}</h1><p>Hover over vertices for copy numbers and '
        body += 'biopsy membership. Hover over edge midpoints for edge information. Labels denote CNP states '
        body += 'and can repeat. An en dash denotes an unlabeled vertex. Vertical placement shows '
        body += 'tree depth, not elapsed time.</p>' + plot
        with (output / f"{key}.html").open("x", encoding="utf-8") as stream:
            stream.write(page(titles[key], body))
    count = len(summary["methods"])
    body = nav + f'<h1>One history, {count} reconstruction{"s" if count != 1 else ""}</h1>'
    body += '<p>This small CTBF v5 example illustrates reconstruction from the same longitudinal biopsies. '
    body += 'Each selected method is identified below. '
    if {"M2", "M4", "M6", "M7"}.intersection(summary["methods"]):
        body += 'Biopsy-guided methods use the shared adaptive-radius/deferred-tie bottom. '
    if {"M2", "M6"} <= summary["methods"].keys():
        body += 'M2 and M6 have the same topology but differ '
        body += 'in whether newly inferred top vertices retain CNP labels.'
    body += '</p>'
    body += f'<p>Seed {summary["seed"]}; {summary["truth_vertices"]} true vertices; '
    body += f'{summary["distinct_sampled_profiles"]} distinct sampled CNPs. '
    body += 'Distances use the minimum of both directed cnp2cnp d_any costs.</p>'
    body += '<table><tr><th>Biopsy</th><th>Generation</th><th>Available</th><th>Sampled</th></tr>'
    for index, row in enumerate(summary["sampling"], 1):
        body += f'<tr><td>{index}</td><td>{row["generation"]}</td><td>{row["available"]}</td><td>{row["selected"]}</td></tr>'
    body += '</table><p>The trimmed truth retains every occurrence of a sampled CNP state and '
    body += 'the unlabeled branching vertices needed to connect them.</p>'
    body += '<table><tr><th>Tree</th><th>Vertices</th><th>GRF, full truth</th><th>GRF, trimmed truth</th><th>AD-F1</th></tr>'
    for key, tree in trees.items():
        row = summary["methods"].get(key, {})
        values = ["—" if row.get(metric) is None else f'{row[metric]:.3f}' for metric in ("grf_full", "grf_trimmed", "ad_f1")]
        if row.get("ad_f1_role") == "partial_diagnostic":
            values[-1] += "*"
        body += f'<tr><td><a href="{key}.html">{escape(titles[key])}</a></td><td>{len(tree)}</td>'
        body += ''.join(f'<td>{v}</td>' for v in values) + '</tr>'
    body += '</table><p>All scores range from 0 to 1, with higher values better. These are illustrative '
    body += 'single-history results. AD-F1 is the closed-state ancestry endpoint. '
    if any(r.get("ad_f1_role") == "partial_diagnostic" for r in summary["methods"].values()):
        body += '* AD-F1 on a partial tree is an optional diagnostic: unlabeled ancestors cannot '
        body += 'contribute labeled ancestor–descendant pairs. Pooled classical NJ has no such pairs. '
    body += 'The AD-F1 convention assigns zero when both relation sets are empty.</p>'
    body += '<p>Download: <a href="config.json">configuration</a> · <a href="input.json">biopsies</a> · '
    body += '<a href="trees.json">trees</a> · <a href="summary.json">scores and provenance</a>.</p>'
    if (output / "true_tree.nwk").is_file():
        body += '<p>Topology-only Newick: ' + ' · '.join(
            f'<a href="{key}.nwk">{escape(key)}</a>' for key in trees) + '.</p>'
    with (output / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(page("CTBF quick-start trees", body))
