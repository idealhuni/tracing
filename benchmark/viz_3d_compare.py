#!/usr/bin/env python3
"""3D interactive comparison of two SWC files using Plotly."""
import sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go


def load_swc(path):
    nodes = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        p = line.split()
        if len(p) < 7:
            continue
        nid = int(p[0])
        nodes[nid] = dict(x=float(p[2]), y=float(p[3]), z=float(p[4]), pid=int(p[6]))
    return nodes


def swc_to_segments(nodes):
    """Return (xs, ys, zs) with None separators for each edge — plotly line format."""
    xs, ys, zs = [], [], []
    for nid, n in nodes.items():
        pid = n['pid']
        if pid == -1 or pid not in nodes:
            continue
        p = nodes[pid]
        xs += [n['x'], p['x'], None]
        ys += [n['y'], p['y'], None]
        zs += [n['z'], p['z'], None]
    return xs, ys, zs


def swc_tips(nodes):
    children = {nid: [] for nid in nodes}
    for nid, n in nodes.items():
        if n['pid'] in children:
            children[n['pid']].append(nid)
    tips = [nid for nid, n in nodes.items()
            if not children.get(nid) and n['pid'] != -1]
    return (
        [nodes[t]['x'] for t in tips],
        [nodes[t]['y'] for t in tips],
        [nodes[t]['z'] for t in tips],
    )


def soma_point(nodes):
    roots = [n for n in nodes.values() if n['pid'] == -1]
    if not roots:
        return None
    r = roots[0]
    return r['x'], r['y'], r['z']


def make_figure(swc_specs, title="SWC 3D Comparison"):
    """
    swc_specs: list of (path, label, color)
    """
    fig = go.Figure()

    for path, label, color in swc_specs:
        if not Path(path).exists():
            print(f"  없음: {path}")
            continue

        nodes = load_swc(path)
        xs, ys, zs = swc_to_segments(nodes)
        tx, ty, tz = swc_tips(nodes)
        soma = soma_point(nodes)

        n_nodes = len(nodes)
        n_tips  = len(tx)
        length  = sum(
            ((nodes[nid]['x']-nodes[nodes[nid]['pid']]['x'])**2 +
             (nodes[nid]['y']-nodes[nodes[nid]['pid']]['y'])**2 +
             (nodes[nid]['z']-nodes[nodes[nid]['pid']]['z'])**2)**0.5
            for nid, n in nodes.items()
            if n['pid'] != -1 and n['pid'] in nodes
        )
        full_label = f"{label}  (nodes={n_nodes:,} tips={n_tips} len={length/1000:.2f}mm)"

        # Neurite edges
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            name=full_label,
            line=dict(color=color, width=2),
            opacity=0.85,
        ))

        # Tips
        fig.add_trace(go.Scatter3d(
            x=tx, y=ty, z=tz,
            mode='markers',
            name=f"{label} tips",
            marker=dict(color=color, size=3, symbol='circle'),
            showlegend=False,
        ))

        # Soma
        if soma:
            fig.add_trace(go.Scatter3d(
                x=[soma[0]], y=[soma[1]], z=[soma[2]],
                mode='markers',
                name=f"{label} soma",
                marker=dict(color=color, size=10, symbol='diamond'),
                showlegend=False,
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        scene=dict(
            xaxis_title='X (µm)',
            yaxis_title='Y (µm)',
            zaxis_title='Z (µm)',
            aspectmode='data',
            bgcolor='#111111',
            xaxis=dict(backgroundcolor='#111111', gridcolor='#333', color='#aaa'),
            yaxis=dict(backgroundcolor='#111111', gridcolor='#333', color='#aaa'),
            zaxis=dict(backgroundcolor='#111111', gridcolor='#333', color='#aaa'),
        ),
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#cccccc'),
        legend=dict(bgcolor='#2a2a2a', bordercolor='#555', borderwidth=1),
        margin=dict(l=0, r=0, t=40, b=0),
        height=800,
    )
    return fig


if __name__ == '__main__':
    ROOT_OUT   = Path('/Users/lee/Tracer/tracer_aniso/output/FN1_01')
    BENCH_OUT  = Path('/Users/lee/Tracer/benchmark/results/ours')

    specs = [
        (ROOT_OUT  / 'neurons_auto.swc',  'root (neurons_auto)',  '#f5c518'),   # 노랑
        (ROOT_OUT  / 'neurons_clean.swc', 'root (neurons_clean)', '#ff8c42'),   # 주황
        (BENCH_OUT / 'FN1_01.swc',        'benchmark',            '#4c9be8'),   # 파랑
    ]

    out_html = Path('/Users/lee/Tracer/benchmark/FN1_01_3d_compare.html')
    fig = make_figure(specs, title='FN1_01 — root vs benchmark (3D)')
    fig.write_html(str(out_html))
    print(f'Saved: {out_html}')
    print('브라우저에서 열어주세요.')
