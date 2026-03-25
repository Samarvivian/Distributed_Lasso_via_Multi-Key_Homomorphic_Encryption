#!/usr/bin/env python3
"""
Plot objective function convergence for TRIAD at different sparsity levels.

Usage:
    cd solve && python plot_sparsity.py
    (or run from solve/build — BUILD_DIR is auto-detected)
"""

import os
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams

# ============================================================================
# IEEE Publication Style — consistent with plot_convergence.py
# ============================================================================
rcParams['text.usetex'] = False
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'Times', 'Palatino', 'Computer Modern Roman']
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 11
rcParams['legend.title_fontsize'] = 12
rcParams['figure.dpi'] = 100
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.1
rcParams['lines.linewidth'] = 2.0
rcParams['lines.markersize'] = 6
rcParams['lines.markeredgewidth'] = 0.5
rcParams['grid.alpha'] = 0.3
rcParams['grid.linestyle'] = '--'
rcParams['grid.linewidth'] = 0.8
rcParams['axes.linewidth'] = 1.0

# ============================================================================
# Config
# ============================================================================
BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')

MARKER_INTERVAL = 5

# 4 sparsity levels: file, display label, color, marker
SPARSITY_SERIES = [
    ('Adaptive_TRIAD_log.csv',  'Sparsity 5%',  '#1f77b4', 'o'),
    ('sparsity_20_log.csv',     'Sparsity 20%', '#2ca02c', 's'),
    ('sparsity_30_log.csv',     'Sparsity 30%', '#ff7f0e', '^'),
    ('sparsity_40_log.csv',     'Sparsity 40%', '#d62728', 'D'),
]

# ============================================================================
# Helpers
# ============================================================================
def load_csv(fname):
    path = os.path.join(BUILD_DIR, fname)
    if not os.path.exists(path):
        print(f'  [WARN] {path} not found, skipping.')
        return None
    if os.path.getsize(path) == 0:
        print(f'  [WARN] {path} is empty, skipping.')
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f'  [WARN] {path} has no data, skipping.')
            return None
        return df
    except Exception as e:
        print(f'  [WARN] Failed to read {path}: {e}')
        return None


def parse_obj_col(series):
    def _conv(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return np.nan
    return np.array([_conv(v) for v in series])


def find_divergence(obj_arr, explosion_threshold=1e4, jump_factor=100, jump_min=100):
    prev = None
    for i, v in enumerate(obj_arr):
        if np.isnan(v) or np.isinf(v):
            return i
        if v > explosion_threshold:
            return i
        if prev is not None and prev > 0 and v > prev * jump_factor and v > jump_min:
            return i
        if not (np.isnan(v) or np.isinf(v)):
            prev = v
    return None


def plot_series(ax, iters, obj_arr, color, marker, lw, ms, label, zorder=2):
    div_idx = find_divergence(obj_arr)

    if div_idx is None:
        ax.plot(iters, obj_arr,
                color=color, linewidth=lw,
                marker=marker, markevery=MARKER_INTERVAL,
                markersize=ms, label=label, zorder=zorder)
        return None

    if div_idx == 0:
        print(f'    {label}: diverged at iter 0, nothing to plot')
        return None

    valid_iters = iters[:div_idx]
    valid_obj   = obj_arr[:div_idx]
    ax.plot(valid_iters, valid_obj,
            color=color, linewidth=lw,
            marker=marker, markevery=MARKER_INTERVAL,
            markersize=ms, label=label, zorder=zorder)
    ax.scatter(valid_iters[-1], valid_obj[-1],
               color=color, marker='X', s=140,
               edgecolors='black', linewidths=0.8,
               zorder=zorder + 5)
    return (valid_iters[-1], valid_obj[-1], div_idx)


# ============================================================================
# Main
# ============================================================================
def main():
    print('=' * 70)
    print('TRIAD Sparsity Convergence Plot')
    print('=' * 70)

    fig, ax = plt.subplots(figsize=(8, 5))
    divergence_annotations = []

    for fname, label, color, marker in SPARSITY_SERIES:
        df = load_csv(fname)
        if df is None:
            continue
        iters = df['iter'].values
        obj   = parse_obj_col(df['objective'])
        res   = plot_series(ax, iters, obj, color, marker,
                            lw=2.0, ms=6, label=label, zorder=2)
        if res is not None:
            x, y, di = res
            divergence_annotations.append(
                (x, y, f'iter {di}, diverged', color))
        fin = obj[~np.isnan(obj)]
        last_str = f'{fin[-1]:.4f}' if len(fin) > 0 else 'N/A'
        print(f'  {label}: {len(iters)} iters, final obj={last_str}')

    # -----------------------------------------------------------------------
    # Axes formatting
    # -----------------------------------------------------------------------
    ax.set_yscale('log')
    ax.set_xlabel('Iteration', fontsize=14, fontweight='normal')
    ax.set_ylabel('Objective (log scale)', fontsize=14, fontweight='normal')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.set_xlim(left=-0.5, right=50)
    ax.set_xticks(np.arange(0, 51, 5))

    # y-limits
    all_valid = []
    for fname, _, _, _ in SPARSITY_SERIES:
        df = load_csv(fname)
        if df is None:
            continue
        obj = parse_obj_col(df['objective'])
        div = find_divergence(obj)
        valid = obj[:div] if div is not None else obj
        all_valid.extend(valid[~np.isnan(valid)].tolist())

    if all_valid:
        ymin = min(all_valid)
        ymax = max(all_valid)
        ax.set_ylim(bottom=ymin * 0.5, top=ymax * 3.0)
    ylim = ax.get_ylim()
    n_iter_total = 50

    # -----------------------------------------------------------------------
    # Divergence annotations
    # -----------------------------------------------------------------------
    for ann_idx, (x, y, text, color) in enumerate(divergence_annotations):
        x_off_right = n_iter_total * 0.04
        if x < n_iter_total * 0.6:
            x_text = x + x_off_right
            ha = 'left'
        else:
            x_text = x - n_iter_total * 0.02
            ha = 'right'
        y_log = np.log10(y)
        log_range = np.log10(ylim[1]) - np.log10(ylim[0])
        y_text_log = y_log + log_range * 0.15 * (1 if ann_idx % 2 == 0 else -1)
        y_text = 10 ** y_text_log
        y_text = max(ylim[0] * 1.05, min(y_text, ylim[1] * 0.82))

        ax.annotate(text,
                    xy=(x, y),
                    xytext=(x_text, y_text),
                    fontsize=11, color='#222222', fontweight='bold',
                    ha=ha,
                    arrowprops=dict(arrowstyle='->',
                                    color=color,
                                    lw=1.5,
                                    shrinkA=6, shrinkB=6,
                                    connectionstyle='arc3,rad=0.1'),
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='#FFFDE7',
                              edgecolor=color,
                              alpha=0.95,
                              linewidth=1.5),
                    zorder=20)

    # -----------------------------------------------------------------------
    # Legend
    # -----------------------------------------------------------------------
    legend = ax.legend(loc='upper right',
                       fontsize=11,
                       ncol=2,
                       columnspacing=1.0,
                       handlelength=1.8,
                       handletextpad=0.8,
                       borderaxespad=0.5,
                       framealpha=0.9,
                       edgecolor='gray',
                       fancybox=True,
                       title='Sparsity Level',
                       title_fontsize=11)

    plt.tight_layout()

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(out_dir, exist_ok=True)

    for fmt in ['pdf', 'png']:
        path = os.path.join(out_dir, f'sparsity_convergence.{fmt}')
        fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight')
        print(f'  Saved: {path}')

    print('\nPlot generation complete.')


if __name__ == '__main__':
    main()
