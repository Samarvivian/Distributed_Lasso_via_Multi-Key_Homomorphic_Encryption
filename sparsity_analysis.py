#!/usr/bin/env python3
"""
TRIAD Sparsity Sensitivity Analysis.

Simulates TRIAD over 5 ground-truth sparsity levels (10%–50%) using the
exact parameters from solve/server.cpp and solve/client.cpp.  Server and
client logic live in server_sim.py and client_sim.py respectively; this
script acts as the orchestrator (no encryption, pure NumPy).

Explosion handling: if objective becomes NaN/Inf or exceeds thresholds the
run for that sparsity level is terminated immediately; all data recorded
*before* the explosion is preserved and plotted.

Usage:
    cd solve && python sparsity_analysis.py
"""

import sys, io, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from server_sim import (
    ServerSim,
    N_FEAT, NUM_PARTIES, LAMBDA, KAPPA,
    MAX_ITER, UPDATE_INTV, SHRINK_WARM,
)
from client_sim import ClientSim

# ============================================================================
# IEEE Publication Style (matches plot_convergence.py)
# ============================================================================
rcParams['text.usetex']           = False
rcParams['font.family']           = 'serif'
rcParams['font.serif']            = ['Times New Roman', 'Times', 'Palatino',
                                      'Computer Modern Roman']
rcParams['font.size']             = 12
rcParams['axes.labelsize']        = 14
rcParams['axes.titlesize']        = 14
rcParams['xtick.labelsize']       = 12
rcParams['ytick.labelsize']       = 12
rcParams['legend.fontsize']       = 11
rcParams['figure.dpi']            = 100
rcParams['savefig.dpi']           = 300
rcParams['savefig.bbox']          = 'tight'
rcParams['lines.linewidth']       = 2.0
rcParams['lines.markersize']      = 6
rcParams['lines.markeredgewidth'] = 0.5
rcParams['grid.alpha']            = 0.3
rcParams['grid.linestyle']        = '--'
rcParams['grid.linewidth']        = 0.8
rcParams['axes.linewidth']        = 1.0
rcParams['axes.axisbelow']        = True
rcParams['axes.grid']             = True
rcParams['legend.framealpha']     = 0.9
rcParams['legend.edgecolor']      = 'gray'
rcParams['legend.fancybox']       = True
rcParams['legend.frameon']        = True
rcParams['legend.handlelength']   = 2.0
rcParams['legend.handletextpad']  = 0.8

# ============================================================================
# Experiment configuration
# ============================================================================
M_ROWS         = 50                        # rows per party
TOTAL_ROWS     = M_ROWS * NUM_PARTIES      # 150
SPARSITY_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]
MARKER_EVERY   = 5

COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
MARKERS = ['o',       's',       'D',       '^',       'v'      ]


# ============================================================================
# Orchestrator: run one TRIAD experiment (server + 3 clients)
# ============================================================================
def run_triad(A_parties: list, b_parties: list,
              A_full: np.ndarray, b_full: np.ndarray,
              x_true: np.ndarray):
    """
    Run one full TRIAD experiment following the server.cpp / client.cpp protocol.

    Protocol outline (mirrors server.cpp main):
        Phase 1  — Bootstrap R: clients send x_i^(0), server computes R^(0)
        Phase 2  — ADMM iterations:
            (a/b) server masked-decrypts v_i = z - u_i → client x-update
            (c/d) server masked-decrypts u_i (transparent in simulation)
            (e)   server aggregates w = (1/K)*sum(x_i + u_i)
            (f)   CRC every UPDATE_INTV iters after SHRINK_WARM: update R
            (g)   Chebyshev z-update
            (h)   clients update u_i = u_i + x_i - z
            (i)   side-decrypt objective; explosion check

    Returns
    -------
    obj_history : list[float]  — valid objectives (length ≤ MAX_ITER)
    R_history   : list[float]  — R values aligned to obj_history
    mse_history : list[float]  — MSE vs x_true, aligned to obj_history
    exploded_at : int | None   — iteration index of explosion, or None
    """
    server  = ServerSim(A_full, b_full)
    clients = [ClientSim(k, A_parties[k], b_parties[k])
               for k in range(NUM_PARTIES)]

    # ---- Phase 1: Bootstrap R -----------------------------------------
    x0_list = [c.phase1_send_x0() for c in clients]
    currentR, coeffs = server.phase1_bootstrap_r(x0_list)

    obj_history: list = []
    R_history:   list = []
    mse_history: list = []
    exploded_at        = None

    # ---- Phase 2: Main ADMM loop --------------------------------------
    for it in range(MAX_ITER):

        # (a/b) Server sends v_i = z - u_i; each client computes x_i update
        x_list = []
        for k in range(NUM_PARTIES):
            v_k = server.prepare_v(clients[k].u)           # masked decrypt sim
            x_k = clients[k].recv_v_do_x_update(v_k)
            x_list.append(x_k)

        # (c/d) Server reads u_i via masked decrypt (transparent in simulation)
        u_list = [c.u.copy() for c in clients]             # snapshot before update

        # (e) Server aggregates w = (1/K)*sum(x_k + u_k)
        w = server.aggregate_w(x_list, u_list)

        # (f) CRC: update R if scheduled
        if it >= SHRINK_WARM and it % UPDATE_INTV == 0:
            R_raw    = server.compute_R_raw(w)
            # Each client checks safe using currentR (not R_raw) — matches CRC fix
            safe_all = all(c.crc_check_safe(R_raw, currentR) for c in clients)
            currentR, coeffs = server.update_R(it, R_raw, currentR, safe_all)

        R_history.append(currentR)

        # (g) Chebyshev z-update
        z = server.z_update(w, currentR, coeffs)

        # (h) Clients update u_i = u_i + x_i - z
        for c in clients:
            c.recv_z_do_u_update(z)

        # (i) Side-decrypt objective + explosion check
        obj      = server.compute_objective(z)
        mse      = float(np.mean((z - x_true) ** 2))
        prev_obj = obj_history[-1] if obj_history else None

        if ServerSim.check_explosion(obj, prev_obj):
            prev_str = f"{prev_obj:.4g}" if prev_obj is not None else "N/A"
            print(f"    [EXPLOSION] iter={it:2d}  obj={obj:.4g}  prev={prev_str}")
            exploded_at = it
            break                     # skip rest; obj_history up to here is valid

        obj_history.append(obj)
        mse_history.append(mse)

    return obj_history, R_history[:len(obj_history)], mse_history, exploded_at


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 65)
    print("TRIAD Sparsity Sensitivity Analysis")
    print(f"  Data : {TOTAL_ROWS}×{N_FEAT}  ({NUM_PARTIES} parties × {M_ROWS} rows)")
    print(f"  λ={LAMBDA}  ρ={1.0}  κ={KAPPA:.4f}  chebyDeg=15")
    print(f"  maxIter={MAX_ITER}  CRC every {UPDATE_INTV} iters after warm-up {SHRINK_WARM}")
    print("=" * 65)

    # Fixed data matrix shared across all sparsity levels
    rng_data = np.random.RandomState(42)
    A_full   = rng_data.randn(TOTAL_ROWS, N_FEAT)
    # Column-normalize (mirrors server.cpp colNormalize)
    norms = np.linalg.norm(A_full, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    A_full /= norms

    A_parties = [A_full[k * M_ROWS:(k + 1) * M_ROWS, :] for k in range(NUM_PARTIES)]

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sparsity_logs')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    results = {}   # label -> dict(obj, R, exploded_at, sparsity)

    for idx, sparsity in enumerate(SPARSITY_LEVELS):
        n_nz  = max(1, int(round(sparsity * N_FEAT)))
        pct   = int(sparsity * 100)
        label = f"Sparsity {pct}%"

        # Ground-truth x* with given sparsity (independent seed per level)
        rng_x   = np.random.RandomState(200 + int(sparsity * 1000))
        x_true  = np.zeros(N_FEAT)
        nz_idx  = rng_x.choice(N_FEAT, n_nz, replace=False)
        x_true[nz_idx] = (rng_x.uniform(0.5, 2.0, n_nz)
                          * rng_x.choice([-1, 1], n_nz))

        b_full    = A_full @ x_true                         # noiseless
        b_parties = [b_full[k * M_ROWS:(k + 1) * M_ROWS] for k in range(NUM_PARTIES)]

        print(f"\n[{idx + 1}/{len(SPARSITY_LEVELS)}] {label}  "
              f"({n_nz}/{N_FEAT} non-zeros,  ‖x*‖₁={np.sum(np.abs(x_true)):.2f})")

        obj_hist, R_hist, mse_hist, exploded_at = run_triad(
            A_parties, b_parties, A_full, b_full, x_true)

        results[label] = dict(
            obj=obj_hist, R=R_hist, mse=mse_hist,
            exploded_at=exploded_at, sparsity=sparsity,
        )

        # Print summary
        if exploded_at is not None:
            last_valid = obj_hist[-1] if obj_hist else float('nan')
            print(f"  EXPLODED at iter {exploded_at}  "
                  f"(last valid obj={last_valid:.6f}, {len(obj_hist)} iters recorded)")
        else:
            print(f"  iter  0: obj={obj_hist[0]:.6f}   R={R_hist[0]:.4f}")
            print(f"  iter {MAX_ITER - 1:2d}: obj={obj_hist[-1]:.6f}   R={R_hist[-1]:.4f}")

        # Save CSV log
        df = pd.DataFrame({
            'iter':        np.arange(len(obj_hist)),
            'objective':   obj_hist,
            'R':           R_hist,
            'mse':         mse_hist,
            'exploded_at': [exploded_at] * len(obj_hist),
        })
        fname = os.path.join(log_dir, f"sparsity_{pct}_triad.csv")
        df.to_csv(fname, index=False)
        print(f"  Saved: {fname}")

    # ================================================================
    # Plot: objective convergence (log scale, IEEE style)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    divergence_annotations = []

    for idx, (label, res) in enumerate(results.items()):
        obj_hist    = res['obj']
        exploded_at = res['exploded_at']
        pct         = int(res['sparsity'] * 100)
        n_nz        = max(1, int(round(res['sparsity'] * N_FEAT)))
        color       = COLORS[idx]
        mk          = MARKERS[idx]
        iters       = np.arange(len(obj_hist))
        curve_label = f"Sparsity {pct}%  ({n_nz} nz)"

        if len(iters) == 0:
            print(f"  [WARN] {label}: no valid data, skipping plot.")
            continue

        ax.plot(iters, obj_hist,
                color=color, marker=mk,
                markevery=MARKER_EVERY, linewidth=2.0, markersize=6,
                label=curve_label, zorder=3 + idx)

        if exploded_at is not None:
            # X marker at last valid point
            ax.scatter(iters[-1], obj_hist[-1],
                       color=color, marker='X', s=160,
                       edgecolors='black', linewidths=0.8,
                       zorder=10 + idx)
            divergence_annotations.append(
                (iters[-1], obj_hist[-1],
                 f"Sparsity {pct}%\niter {exploded_at}, exploded",
                 color))

    # Axes
    ax.set_yscale('log')
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Objective (log scale)', fontsize=14)
    ax.set_title('TRIAD Convergence vs. Ground-Truth Sparsity', fontsize=14)
    ax.set_xlim(left=-0.5, right=MAX_ITER - 0.5)
    ax.set_xticks(np.arange(0, MAX_ITER + 1, 5))
    ax.grid(True, which='both', linestyle='--', alpha=0.3)

    # Y-limits from finite positive valid values
    all_vals = [v for res in results.values()
                for v in res['obj'] if np.isfinite(v) and v > 0]
    if all_vals:
        ax.set_ylim(bottom=min(all_vals) * 0.7, top=max(all_vals) * 1.5)

    # Explosion annotations (clamped inside plot)
    ylim = ax.get_ylim()
    for i, (x, y, text, color) in enumerate(divergence_annotations):
        y_text = max(ylim[0] * 1.05,
                     min(y * (2.5 + i * 1.5), ylim[1] * 0.82))
        x_off  = MAX_ITER * (0.04 + i * 0.02)
        if x + x_off < MAX_ITER - 2:
            x_text, ha = x + x_off, 'left'
        else:
            x_text, ha = x - MAX_ITER * 0.02, 'right'

        ax.annotate(text,
                    xy=(x, y), xytext=(x_text, y_text),
                    fontsize=11, color='#222222', fontweight='bold', ha=ha,
                    arrowprops=dict(arrowstyle='->',
                                    color=color, lw=1.5,
                                    shrinkA=6, shrinkB=6,
                                    connectionstyle='arc3,rad=0.1'),
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='#FFFDE7',
                              edgecolor=color, alpha=0.95, linewidth=1.5),
                    zorder=20)

    ax.legend(loc='upper right', fontsize=10, ncol=1,
              framealpha=0.9, edgecolor='gray', fancybox=True)
    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        path = os.path.join(out_dir, f'triad_sparsity_sensitivity.{fmt}')
        fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {path}")

    print("\nAll done.")
    print(f"  Logs → solve/sparsity_logs/sparsity_XX_triad.csv")
    print(f"  Figs → solve/figures/triad_sparsity_sensitivity.{{pdf,png}}")


if __name__ == '__main__':
    main()
