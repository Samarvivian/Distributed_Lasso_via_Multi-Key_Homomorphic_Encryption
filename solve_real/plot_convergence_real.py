#!/usr/bin/env python3
"""
Plot objective function convergence for Riboflavin real-data experiment:
PlainADMM vs Static-R=2.0 vs Adaptive TRIAD.

Usage:
    cd solve_real && python plot_convergence_real.py
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
# Style — matches solve/plot_convergence.py exactly
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
rcParams['savefig.format'] = 'pdf'

rcParams['lines.linewidth'] = 2.0
rcParams['lines.markersize'] = 6
rcParams['lines.markeredgewidth'] = 0.5

rcParams['grid.alpha'] = 0.3
rcParams['grid.linestyle'] = '--'
rcParams['grid.linewidth'] = 0.8

rcParams['axes.linewidth'] = 1.0
rcParams['axes.axisbelow'] = True
rcParams['axes.grid'] = True

rcParams['legend.framealpha'] = 0.9
rcParams['legend.edgecolor'] = 'gray'
rcParams['legend.fancybox'] = True
rcParams['legend.frameon'] = True
rcParams['legend.borderaxespad'] = 0.5
rcParams['legend.handlelength'] = 2.0
rcParams['legend.handletextpad'] = 0.8

COLORS = {
    'plaintext': '#1f77b4',   # Blue
    'static':    '#d62728',   # Red
    'triad':     '#2ca02c',   # Green
}

MARKER_INTERVAL = 10  # sparser for 100 iterations

# ============================================================================
# Data directory
# ============================================================================
BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')


# ============================================================================
# Helpers — identical to solve/plot_convergence.py
# ============================================================================
def load_csv(fname):
    path = os.path.join(BUILD_DIR, fname)
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found, skipping.")
        return None
    if os.path.getsize(path) == 0:
        print(f"  [WARN] {path} is empty, skipping.")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or len(df.columns) == 0:
            print(f"  [WARN] {path} has no data, skipping.")
            return None
        return df
    except Exception as e:
        print(f"  [WARN] Failed to read {path}: {e}")
        return None


def parse_col(series):
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
        print(f"    {label}: diverged at iter 0, nothing to plot")
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


def annotate_divergences(ax, divergence_annotations, n_iter_total=100):
    x_max = n_iter_total - 1
    ylim  = ax.get_ylim()

    for i, (x, y, text, color) in enumerate(divergence_annotations):
        y_text = y * (2.0 + i * 1.2)
        x_off  = n_iter_total * (0.04 + i * 0.02)
        if x + x_off < x_max - 2:
            x_text, ha = x + x_off, 'left'
        else:
            x_text, ha = x - n_iter_total * 0.02, 'right'

        y_text = max(ylim[0] * 1.05, min(y_text, ylim[1] * 0.82))

        ax.annotate(text,
                    xy=(x, y),
                    xytext=(x_text, y_text),
                    fontsize=11, color='#222222', fontweight='bold',
                    ha=ha,
                    arrowprops=dict(arrowstyle='->',
                                    color=color, lw=1.5,
                                    shrinkA=6, shrinkB=6,
                                    connectionstyle='arc3,rad=0.1'),
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='#FFFDE7', edgecolor=color,
                              alpha=0.95, linewidth=1.5),
                    zorder=20)


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("TRIAD Convergence Plot  (Riboflavin: PlainADMM / Static-R=2.0 / TRIAD)")
    print("=" * 70)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    div_annotations = []
    n_iter_total = 100

    # ------------------------------------------------------------------
    # 1) Plaintext ADMM
    # ------------------------------------------------------------------
    df = load_csv('plaintext_admm_real_log.csv')
    if df is not None:
        iters = df['iter'].values
        obj   = parse_col(df['objective'])

        res = plot_series(ax, iters, obj,
                          color=COLORS['plaintext'], marker='o',
                          lw=2.0, ms=6, label='Plaintext ADMM', zorder=4)
        if res:
            x, y, di = res
            div_annotations.append((x, y, f'Static R=2.0,iter {di}, diverged', COLORS['plaintext']))

        fin_obj = obj[~np.isnan(obj)]
        if len(fin_obj) > 0:
            print(f"  PlainADMM: {len(iters)} iters, final obj={fin_obj[-1]:.6f}")

    # ------------------------------------------------------------------
    # 2) Static-R = 2.0
    # ------------------------------------------------------------------
    df = load_csv('StaticR_2_0_real_log.csv')
    if df is not None:
        iters = df['iter'].values
        obj   = parse_col(df['objective'])

        res = plot_series(ax, iters, obj,
                          color=COLORS['static'], marker='s',
                          lw=1.8, ms=6, label='Static R=2.0', zorder=3)
        if res:
            x, y, di = res
            div_annotations.append((x, y, f'iter {di}, diverged', COLORS['static']))

        fin_obj = obj[~np.isnan(obj) & (obj > 0)]
        if len(fin_obj) > 0:
            print(f"  Static R=2.0: {len(iters)} iters, last valid obj={fin_obj[-1]:.6f}")

    # ------------------------------------------------------------------
    # 3) Adaptive TRIAD
    # ------------------------------------------------------------------
    df = load_csv('Adaptive_TRIAD_real_log.csv')
    if df is not None:
        iters = df['iter'].values
        obj   = parse_col(df['objective'])

        res = plot_series(ax, iters, obj,
                          color=COLORS['triad'], marker='^',
                          lw=2.5, ms=7, label='Adaptive TRIAD (Ours)', zorder=5)
        if res:
            x, y, di = res
            div_annotations.append((x, y, f'iter {di}, diverged', COLORS['triad']))

        fin_obj = obj[~np.isnan(obj)]
        if len(fin_obj) > 0:
            print(f"  Adaptive TRIAD: {len(iters)} iters, final obj={fin_obj[-1]:.6f}")

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------
    ax.set_yscale('log')
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Objective (log scale)', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')

    ax.set_xlim(left=-1, right=n_iter_total)
    ax.set_xticks(np.arange(0, n_iter_total + 1, 10))

    # y-limits
    all_valid_obj = []
    for fname in ['plaintext_admm_real_log.csv',
                  'StaticR_2_0_real_log.csv',
                  'Adaptive_TRIAD_real_log.csv']:
        df2 = load_csv(fname)
        if df2 is None:
            continue
        col = 'objective' if 'objective' in df2.columns else df2.columns[-2]
        obj2 = parse_col(df2[col])
        div2 = find_divergence(obj2)
        valid = obj2[:div2] if div2 is not None else obj2
        valid = valid[(~np.isnan(valid)) & (valid > 0)]
        if len(valid):
            all_valid_obj.extend(valid.tolist())
    if all_valid_obj:
        ax.set_ylim(bottom=min(all_valid_obj) * 0.7, top=max(all_valid_obj) * 1.5)

    annotate_divergences(ax, div_annotations, n_iter_total)

    ax.legend(loc='upper right', fontsize=11, ncol=1,
              handlelength=1.8, handletextpad=0.8,
              borderaxespad=0.5, framealpha=0.9,
              edgecolor='gray', fancybox=True)

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    for fmt in ['pdf', 'png']:
        path = os.path.join(out_dir, f'triad_real_convergence.{fmt}')
        fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
