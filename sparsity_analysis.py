#!/usr/bin/env python3
"""
TRIAD Sparsity Sensitivity Analysis (solve/ experiment setting).

Simulates TRIAD with 5 different ground-truth sparsity levels (10%–50%)
using the exact parameters from solve/server.cpp and solve/client.cpp:
  A: 150×200 (3 parties × 50 rows), column-normalized Gaussian
  rho=1.0, lambda=0.1, maxIter=50, chebyDeg=15
  CRC: alpha=1.2, updateInterval=5, shrinkWarmup=5, delta_safe=0.95, gamma_smooth=0.8

Usage:
    cd solve && python sparsity_analysis.py
"""

import sys, io, os

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ============================================================================
# IEEE Publication Style (matches plot_convergence.py)
# ============================================================================
rcParams['text.usetex']        = False
rcParams['font.family']        = 'serif'
rcParams['font.serif']         = ['Times New Roman', 'Times', 'Palatino',
                                   'Computer Modern Roman']
rcParams['font.size']          = 12
rcParams['axes.labelsize']     = 14
rcParams['axes.titlesize']     = 14
rcParams['xtick.labelsize']    = 12
rcParams['ytick.labelsize']    = 12
rcParams['legend.fontsize']    = 11
rcParams['figure.dpi']         = 100
rcParams['savefig.dpi']        = 300
rcParams['savefig.bbox']       = 'tight'
rcParams['lines.linewidth']    = 2.0
rcParams['lines.markersize']   = 6
rcParams['lines.markeredgewidth'] = 0.5
rcParams['grid.alpha']         = 0.3
rcParams['grid.linestyle']     = '--'
rcParams['grid.linewidth']     = 0.8
rcParams['axes.linewidth']     = 1.0
rcParams['axes.axisbelow']     = True
rcParams['axes.grid']          = True
rcParams['legend.framealpha']  = 0.9
rcParams['legend.edgecolor']   = 'gray'
rcParams['legend.fancybox']    = True
rcParams['legend.frameon']     = True
rcParams['legend.handlelength']= 2.0
rcParams['legend.handletextpad']= 0.8

# ============================================================================
# Parameters — exact match to solve/server.cpp and solve/client.cpp
# ============================================================================
N_FEAT        = 200
M_ROWS        = 50      # rows per party (server.cpp: M_ROWS = 50)
NUM_PARTIES   = 3
TOTAL_ROWS    = M_ROWS * NUM_PARTIES   # = 150

RHO           = 1.0
LAMBDA        = 0.1
KAPPA         = LAMBDA / (RHO * NUM_PARTIES)  # 0.0333... (server.cpp: kappa = lambda/(rho*K))

MAX_ITER      = 50
UPDATE_INTV   = 5       # CRC every N iterations
SHRINK_WARM   = 5       # CRC starts at iter SHRINK_WARM
CHEBY_DEG     = 15

ALPHA_CRC     = 1.2     # server.cpp: alpha_crc = 1.2
GAMMA_SMOOTH  = 0.8
DELTA_SAFE    = 0.95

SPARSITY_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]
MARKER_EVERY  = 5

# One color/marker per sparsity level
COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
MARKERS = ['o', 's', 'D', '^', 'v']


# ============================================================================
# Chebyshev soft-threshold approximation (matches client.cpp chebyCoeffs)
# ============================================================================
def soft_threshold(w, kappa):
    return np.sign(w) * np.maximum(np.abs(w) - kappa, 0.0)


def cheby_coeffs(R, deg, kappa):
    k = np.arange(deg + 1)
    nodes = np.cos(np.pi * (k + 0.5) / (deg + 1))
    f_nodes = soft_threshold(nodes * R, kappa)
    coeffs = np.zeros(deg + 1)
    for j in range(deg + 1):
        weights = np.cos(np.pi * j * (k + 0.5) / (deg + 1))
        coeffs[j] = (2.0 / (deg + 1)) * np.sum(f_nodes * weights)
    return coeffs


def eval_cheby(w, coeffs, R):
    w_norm = np.clip(w / R, -1.0, 1.0)
    d = len(coeffs) - 1
    T = [np.ones_like(w_norm), w_norm.copy()]
    for _ in range(2, d + 1):
        T.append(2.0 * w_norm * T[-1] - T[-2])
    y = np.zeros_like(w)
    for j in range(d + 1):
        y += coeffs[j] * T[j]
    return y - coeffs[0] / 2.0


# ============================================================================
# Objective: (1/2)||Az - b||^2 + lambda * ||z||_1  (matches computeObjective)
# ============================================================================
def objective(z, A, b):
    r = A @ z - b
    return 0.5 * np.dot(r, r) + LAMBDA * np.sum(np.abs(z))


# ============================================================================
# TRIAD simulation — distributed ADMM with 3 parties + CRC adaptation
# Mirrors server.cpp Phase1 (Bootstrap-R) + Phase2 (main loop)
# ============================================================================
def run_triad(A_parties, b_parties, A_full, b_full):
    """
    A_parties: list of 3 arrays, each (M_ROWS, N_FEAT)
    b_parties: list of 3 vectors, each (M_ROWS,)
    A_full, b_full: stacked matrix/vector for objective computation
    Returns: (obj_history, R_history)  each of length MAX_ITER
    """
    n = N_FEAT

    # Precompute per-party x-update inverses: M_k = (A_k^T A_k + rho I)^{-1}
    M_inv = []
    g_vec = []
    for k in range(NUM_PARTIES):
        Ak, bk = A_parties[k], b_parties[k]
        Mk = np.linalg.solve(Ak.T @ Ak + RHO * np.eye(n), np.eye(n))
        M_inv.append(Mk)
        g_vec.append(Ak.T @ bk)

    # Initialize
    z = np.zeros(n)
    u = [np.zeros(n) for _ in range(NUM_PARTIES)]
    x = [np.zeros(n) for _ in range(NUM_PARTIES)]

    # --- Phase 1: Bootstrap R ---
    # x_bar = mean_k { M_k @ g_k }  (z=0, u=0 initial estimate)
    x_bar = np.zeros(n)
    for k in range(NUM_PARTIES):
        x_bar += M_inv[k] @ g_vec[k]
    x_bar /= NUM_PARTIES

    sumSq = np.dot(x_bar, x_bar)
    currentR = max(1.5 * np.sqrt(sumSq / n), 3.0 * KAPPA, 1.5)
    coeffs = cheby_coeffs(currentR, CHEBY_DEG, KAPPA)

    obj_history = []
    R_history   = []

    # --- Phase 2: Main ADMM loop ---
    for it in range(MAX_ITER):
        # x-update: x_k = M_k @ (g_k + rho * (z - u_k))
        for k in range(NUM_PARTIES):
            x[k] = M_inv[k] @ (g_vec[k] + RHO * (z - u[k]))

        # Consensus w = (1/K) * sum(x_k + u_k)
        w = np.zeros(n)
        for k in range(NUM_PARTIES):
            w += x[k] + u[k]
        w /= NUM_PARTIES

        # CRC: update R  (mirrors server.cpp Phase 2 CRC block)
        if it >= SHRINK_WARM and it % UPDATE_INTV == 0:
            Psi  = np.dot(w, w)                              # ||w||^2
            R_raw = ALPHA_CRC * np.sqrt(Psi / n)
            maxW  = np.max(np.abs(w))
            safe  = (maxW <= DELTA_SAFE * currentR)

            if R_raw > currentR:
                currentR = R_raw
            elif not safe:
                currentR = currentR / DELTA_SAFE
            elif it > SHRINK_WARM and safe:
                currentR = max(R_raw, GAMMA_SMOOTH * currentR)

            coeffs = cheby_coeffs(currentR, CHEBY_DEG, KAPPA)

        R_history.append(currentR)

        # z-update: Chebyshev soft-threshold
        z = eval_cheby(w, coeffs, currentR)

        # u-update: u_k = u_k + x_k - z
        for k in range(NUM_PARTIES):
            u[k] = u[k] + x[k] - z

        obj_history.append(objective(z, A_full, b_full))

    return obj_history, R_history


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 65)
    print("TRIAD Sparsity Sensitivity Analysis")
    print(f"  Data: {TOTAL_ROWS}x{N_FEAT} ({NUM_PARTIES} parties x {M_ROWS} rows)")
    print(f"  lambda={LAMBDA}, rho={RHO}, kappa={KAPPA:.4f}, chebyDeg={CHEBY_DEG}")
    print(f"  maxIter={MAX_ITER}, alpha_crc={ALPHA_CRC}, CRC every {UPDATE_INTV} iters")
    print("=" * 65)

    # Fixed random data matrix (same A for all sparsity levels)
    rng_data = np.random.RandomState(42)
    A_full = rng_data.randn(TOTAL_ROWS, N_FEAT)
    # Column normalize (matches server.cpp)
    col_norms = np.linalg.norm(A_full, axis=0, keepdims=True)
    col_norms[col_norms == 0] = 1.0
    A_full = A_full / col_norms

    # Split into party data
    A_parties = [A_full[k*M_ROWS:(k+1)*M_ROWS, :] for k in range(NUM_PARTIES)]

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sparsity_logs')
    os.makedirs(log_dir, exist_ok=True)

    all_obj = {}   # label -> list of floats
    all_R   = {}

    for idx, sparsity in enumerate(SPARSITY_LEVELS):
        n_nz  = max(1, int(round(sparsity * N_FEAT)))
        label = f"Sparsity {int(sparsity*100)}%"

        # Generate x_true with given sparsity (independent seed per level)
        rng_x = np.random.RandomState(200 + int(sparsity * 1000))
        x_true = np.zeros(N_FEAT)
        nz_idx = rng_x.choice(N_FEAT, n_nz, replace=False)
        x_true[nz_idx] = (rng_x.uniform(0.5, 2.0, n_nz)
                          * rng_x.choice([-1, 1], n_nz))

        b_full = A_full @ x_true          # no noise (matches server.cpp)
        b_parties = [b_full[k*M_ROWS:(k+1)*M_ROWS] for k in range(NUM_PARTIES)]

        print(f"\n[{idx+1}/5] {label}  ({n_nz}/{N_FEAT} non-zeros, "
              f"||x*||_1={np.sum(np.abs(x_true)):.2f})")

        obj_hist, R_hist = run_triad(A_parties, b_parties, A_full, b_full)

        all_obj[label] = obj_hist
        all_R[label]   = R_hist

        # Save CSV log
        df = pd.DataFrame({
            'iter':      np.arange(MAX_ITER),
            'objective': obj_hist,
            'R':         R_hist,
        })
        fname = os.path.join(log_dir, f"sparsity_{int(sparsity*100)}_triad.csv")
        df.to_csv(fname, index=False)
        print(f"  iter  0: obj={obj_hist[0]:.6f}   R={R_hist[0]:.4f}")
        print(f"  iter {MAX_ITER-1:2d}: obj={obj_hist[-1]:.6f}   R={R_hist[-1]:.4f}")
        print(f"  Saved: {fname}")

    # ============================================================================
    # Plot: objective convergence (log scale, IEEE style, one curve per sparsity)
    # ============================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    iters = np.arange(MAX_ITER)
    for idx, (label, obj_hist) in enumerate(all_obj.items()):
        pct = int(SPARSITY_LEVELS[idx] * 100)
        ax.plot(iters, obj_hist,
                color=COLORS[idx], marker=MARKERS[idx],
                markevery=MARKER_EVERY, linewidth=2.0, markersize=6,
                label=f'Sparsity {pct}%  ({int(SPARSITY_LEVELS[idx]*N_FEAT)} nz)',
                zorder=3 + idx)

    ax.set_yscale('log')
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Objective (log scale)', fontsize=14)
    ax.set_title('TRIAD Convergence vs. Ground-Truth Sparsity', fontsize=14)
    ax.set_xlim(left=-0.5, right=MAX_ITER - 0.5)
    ax.set_xticks(np.arange(0, MAX_ITER + 1, 5))
    ax.grid(True, which='both', linestyle='--', alpha=0.3)

    # y-axis limits from finite positive values
    all_vals = [v for lst in all_obj.values()
                for v in lst if np.isfinite(v) and v > 0]
    if all_vals:
        ax.set_ylim(bottom=min(all_vals) * 0.7, top=max(all_vals) * 1.5)

    ax.legend(loc='upper right', fontsize=10, ncol=1,
              framealpha=0.9, edgecolor='gray', fancybox=True)

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        path = os.path.join(out_dir, f'triad_sparsity_sensitivity.{fmt}')
        fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {path}")

    print("\nAll done.")
    print(f"  Logs  → solve/sparsity_logs/sparsity_XX_triad.csv")
    print(f"  Figs  → solve/figures/triad_sparsity_sensitivity.{{pdf,png}}")


if __name__ == '__main__':
    main()
