#!/usr/bin/env python3
"""
Plot objective function convergence for PlainADMM, Static-R, and Adaptive TRIAD.

Usage:
    cd solve && python plot_convergence.py
    (or run from solve/build, adjust BUILD_DIR below)
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
# IEEE Publication Style - Original marker styles, larger fonts
# ============================================================================
# Set LaTeX rendering for professional appearance
rcParams['text.usetex'] = False  # Set to True if you have LaTeX installed
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'Times', 'Palatino', 'Computer Modern Roman']
rcParams['font.size'] = 12  # Base font size increased from 10
rcParams['axes.labelsize'] = 14  # Axis labels larger (was 11)
rcParams['axes.titlesize'] = 14  # Title size (was 12)
rcParams['xtick.labelsize'] = 12  # Tick labels larger (was 10)
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 11  # Legend text larger (was 9)
rcParams['legend.title_fontsize'] = 12

# Figure settings for high-quality output
rcParams['figure.dpi'] = 100
rcParams['savefig.dpi'] = 300  # High DPI for publication
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.1
rcParams['savefig.format'] = 'pdf'

# Line and marker settings - keeping original style
rcParams['lines.linewidth'] = 2.0
rcParams['lines.markersize'] = 6
rcParams['lines.markeredgewidth'] = 0.5

# Grid settings - original style
rcParams['grid.alpha'] = 0.3
rcParams['grid.linestyle'] = '--'
rcParams['grid.linewidth'] = 0.8

# Axes settings
rcParams['axes.linewidth'] = 1.0
rcParams['axes.axisbelow'] = True
rcParams['axes.grid'] = True

# Legend settings - original style
rcParams['legend.framealpha'] = 0.9
rcParams['legend.edgecolor'] = 'gray'
rcParams['legend.fancybox'] = True
rcParams['legend.frameon'] = True
rcParams['legend.borderaxespad'] = 0.5
rcParams['legend.handlelength'] = 2.0
rcParams['legend.handletextpad'] = 0.8

# Color palette - keeping original colors
COLORS = {
    'plaintext': '#1f77b4',  # Blue
    'crc': '#2ca02c',  # Green
    'static': {  # Original colors
        0.5: '#d62728',  # Red
        1.5: '#ff7f0e',  # Orange
        2.0: '#17becf',  # Cyan
        3.0: '#8c564b',  # Brown
        5.0: '#9467bd',  # Purple
        10.0: '#e377c2',  # Pink
    }
}

# Original marker interval
MARKER_INTERVAL = 5

# ============================================================================
# Data directory
# ============================================================================
BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')


# ============================================================================
# Helpers (unchanged)
# ============================================================================
def load_csv(fname):
    """Load CSV file with error handling."""
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


def parse_obj_col(series):
    """Convert objective column to float, treating 'ABORT'/'FAIL'/'NaN' as np.nan."""

    def _conv(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return np.nan

    return np.array([_conv(v) for v in series])


def find_divergence(obj_arr, explosion_threshold=1e4, jump_factor=100, jump_min=100):
    """
    Return the index of the first diverged point (1-based annotation).
    Criteria:
      - value is NaN/Inf
      - value > explosion_threshold
      - value > prev * jump_factor AND value > jump_min
    Returns None if no divergence found.
    """
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
    """Plot a single algorithm's objective curve."""
    div_idx = find_divergence(obj_arr)

    if div_idx is None:
        # No divergence — plot everything
        ax.plot(iters, obj_arr,
                color=color, linewidth=lw,
                marker=marker, markevery=MARKER_INTERVAL,
                markersize=ms, label=label, zorder=zorder)
        return None  # no divergence annotation needed

    if div_idx == 0:
        print(f"    {label}: diverged at iter 0, nothing to plot")
        return None

    # Plot valid portion
    valid_iters = iters[:div_idx]
    valid_obj = obj_arr[:div_idx]
    ax.plot(valid_iters, valid_obj,
            color=color, linewidth=lw,
            marker=marker, markevery=MARKER_INTERVAL,
            markersize=ms, label=label, zorder=zorder)

    # X mark at last valid point
    ax.scatter(valid_iters[-1], valid_obj[-1],
               color=color, marker='X', s=140,
               edgecolors='black', linewidths=0.8,
               zorder=zorder + 5)

    return (valid_iters[-1], valid_obj[-1], div_idx)  # for annotation


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("TRIAD Convergence Plot  (PlainADMM + Static-R + Adaptive TRIAD)")
    print("=" * 70)

    # Create figure with IEEE recommended size
    fig, ax = plt.subplots(figsize=(8, 5))  # Slightly larger for better readability

    divergence_annotations = []  # list of (x, y, text, color)

    # ------------------------------------------------------------------
    # 1) Plaintext ADMM - original marker style (circle)
    # ------------------------------------------------------------------
    df = load_csv('plaintext_admm_log.csv')
    if df is not None:
        iters = df['iter'].values
        obj = parse_obj_col(df['objective'])
        res = plot_series(ax, iters, obj,
                          color=COLORS['plaintext'], marker='o',  # Circle (original)
                          lw=2.0, ms=6,
                          label='Plaintext ADMM', zorder=4)
        if res:
            x, y, di = res
            divergence_annotations.append(
                (x, y, f'iter {di}, diverged', COLORS['plaintext']))
        print(f"  PlainADMM: {len(iters)} iters, "
              f"final obj={obj[~np.isnan(obj)][-1]:.4f}")

    # ------------------------------------------------------------------
    # 2) Static-R experiments - original markers
    # ------------------------------------------------------------------
    static_Rs = [
        (0.5, 's', 1.5),  # (R_value, marker, linewidth) - square (original)
        (1.5, 'D', 1.5),  # diamond (original)
        (2.0, 'p', 1.5),  # pentagon (original)
        (3.0, '*', 1.8),  # star (original)
        (5.0, 'v', 1.5),  # triangle down (original)
        (10.0, '^', 1.5),  # triangle up (original)
    ]
    for R, mk, lw in static_Rs:
        r_str = f'{R:.1f}'.replace('.', '_')
        fname = f'StaticR_{r_str}_log.csv'
        df = load_csv(fname)
        if df is None:
            continue

        iters = df['iter'].values
        obj = parse_obj_col(df['objective'])
        color = COLORS['static'][R]
        label = f'Static R={R}'

        res = plot_series(ax, iters, obj,
                          color=color, marker=mk,
                          lw=lw, ms=6, label=label, zorder=2)
        if res:
            x, y, di = res
            divergence_annotations.append(
                (x, y, f'R={R}, iter {di}, diverged', color))  # Restored full text
        fin = obj[~np.isnan(obj)]
        last_str = f'{fin[-1]:.4f}' if len(fin) > 0 else 'N/A'
        print(f"  Static R={R}: {len(iters)} rows, "
              f"div={find_divergence(obj)}, "
              f"last valid={last_str}")

    # ------------------------------------------------------------------
    # 3) Adaptive TRIAD - original marker (triangle up)
    # ------------------------------------------------------------------
    df = load_csv('Adaptive_TRIAD_log.csv')
    if df is not None:
        iters = df['iter'].values
        obj = parse_obj_col(df['objective'])
        res = plot_series(ax, iters, obj,
                          color=COLORS['crc'], marker='^',  # Triangle up (original)
                          lw=2.5, ms=7,
                          label='Adaptive TRIAD (Ours)', zorder=5)
        if res:
            x, y, di = res
            divergence_annotations.append(
                (x, y, f'iter {di}, diverged', COLORS['crc']))
        fin = obj[~np.isnan(obj)]
        last_str = f'{fin[-1]:.4f}' if len(fin) > 0 else 'N/A'
        print(f"  Adaptive TRIAD: {len(iters)} iters, "
              f"final obj={last_str}")

    # ------------------------------------------------------------------
    # Axes formatting - larger labels
    # ------------------------------------------------------------------
    ax.set_yscale('log')
    ax.set_xlabel('Iteration', fontsize=14, fontweight='normal')
    ax.set_ylabel('Objective (log scale)', fontsize=14, fontweight='normal')

    # Grid - original style
    ax.grid(True, alpha=0.3, linestyle='--', which='both')

    ax.set_xlim(left=-0.5, right=50)
    ax.set_xticks(np.arange(0, 51, 5))

    # Determine y-limits from valid data
    all_valid = []
    for fname_pat in ['plaintext_admm_log.csv',
                      'StaticR_0_5_log.csv', 'StaticR_1_5_log.csv',
                      'StaticR_2_0_log.csv', 'StaticR_3_0_log.csv',
                      'StaticR_5_0_log.csv', 'StaticR_10_0_log.csv',
                      'Adaptive_TRIAD_log.csv']:
        df2 = load_csv(fname_pat)
        if df2 is None:
            continue
        col = 'objective' if 'objective' in df2.columns else df2.columns[-2]
        obj2 = parse_obj_col(df2[col])
        div2 = find_divergence(obj2)
        valid = obj2[:div2] if div2 is not None else obj2
        valid = valid[(~np.isnan(valid)) & (valid > 0)]
        if len(valid):
            all_valid.extend(valid.tolist())

    if all_valid:
        y_min = min(all_valid) * 0.7
        y_max = max(all_valid) * 1.5
        ax.set_ylim(bottom=y_min, top=y_max)

    # ------------------------------------------------------------------
    # Divergence annotations - SHORTER ARROWS (keeping the shortened version)
    # ------------------------------------------------------------------
    n_iter_total = 50
    x_max = 49
    ylim = ax.get_ylim()

    for i, (x, y, text, color) in enumerate(divergence_annotations):
        # Reduced multiplier from 4.0 to 2.0 for shorter arrows
        y_offset_factor = 2.0 + i * 1.2  # Was 4.0 + i * 2.5
        y_text = y * y_offset_factor

        # Reduced horizontal offset
        x_off_right = n_iter_total * (0.04 + i * 0.02)  # Was 0.08 + i * 0.04
        if x + x_off_right < x_max - 2:
            x_text = x + x_off_right
            ha = 'left'
        else:
            x_text = x - n_iter_total * 0.02  # Was 0.04
            ha = 'right'

        # Clamp within y-axis bounds
        y_text = max(ylim[0] * 1.05, min(y_text, ylim[1] * 0.9))  # Increased bottom margin

        # Shorter arrow with less shrink
        ax.annotate(text,
                    xy=(x, y),
                    xytext=(x_text, y_text),
                    fontsize=9, color='#222222', fontweight='bold',
                    ha=ha,
                    arrowprops=dict(arrowstyle='->',
                                    color=color,
                                    lw=1.0,  # Slightly thinner arrow
                                    shrinkA=5,  # Increased from 0 to create gap from point
                                    shrinkB=5,  # Increased from 3 to create gap from text
                                    connectionstyle='arc3,rad=0.1'),  # Slight curve
                    bbox=dict(boxstyle='round,pad=0.2',  # Smaller padding
                              facecolor='#FFFDE7',
                              edgecolor=color,
                              alpha=0.95,
                              linewidth=1.0),  # Thinner border
                    zorder=20)

    # ------------------------------------------------------------------
    # Legend - original style with two columns, larger font
    # ------------------------------------------------------------------
    legend = ax.legend(loc='upper right',
                       fontsize=11,  # Larger font
                       ncol=2,
                       columnspacing=1.0,
                       handlelength=1.8,
                       handletextpad=0.8,
                       borderaxespad=0.5,
                       framealpha=0.9,
                       edgecolor='gray',
                       fancybox=True)

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save - IEEE preferred formats
    # ------------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(out_dir, exist_ok=True)

    for fmt in ['pdf', 'png']:
        path = os.path.join(out_dir, f'triad_convergence.{fmt}')
        fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")

    print("\nPlot generation complete. Figures saved in 'figures' directory.")


if __name__ == '__main__':
    main()