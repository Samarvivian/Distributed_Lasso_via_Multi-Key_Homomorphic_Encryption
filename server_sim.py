
"""
server_sim.py — TRIAD server-side simulation (mirrors server.cpp).

Contains the ServerSim class plus Chebyshev math helpers.
The server manages global state (z, Chebyshev coefficients) and drives each
iteration: Bootstrap-R, CRC R-update, Chebyshev z-update, objective monitoring.
"""

import numpy as np

# ============================================================================
# Parameters — must match client_sim.py and solve/server.cpp
# ============================================================================
N_FEAT       = 200
NUM_PARTIES  = 3
RHO          = 1.0
LAMBDA       = 0.1
KAPPA        = LAMBDA / (RHO * NUM_PARTIES)   # = 0.03333...

MAX_ITER     = 50
UPDATE_INTV  = 5      # CRC every N iters
SHRINK_WARM  = 5      # CRC starts at this iteration
CHEBY_DEG    = 15

ALPHA_CRC    = 1.2
GAMMA_SMOOTH = 0.8
DELTA_SAFE   = 0.95

# Explosion-detection thresholds
EXPLOSION_ABS   = 1e6      # obj > this → exploded
EXPLOSION_JUMP  = 100.0    # obj > prev * this AND obj > EXPLOSION_MIN → exploded
EXPLOSION_MIN   = 100.0


# ============================================================================
# Chebyshev helpers (mirror server.cpp chebyCoeffs + EvalChebyshevSeries)
# ============================================================================
def _soft_threshold(x: np.ndarray, kappa: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - kappa, 0.0)


def cheby_coeffs(R: float, deg: int = CHEBY_DEG, kappa: float = KAPPA) -> np.ndarray:
    """
    Compute Chebyshev coefficients for soft-threshold on [-R, R].
    c[0] is pre-divided by 2 (mirrors server.cpp `c[0] /= 2.0`).
    """
    M = deg + 1
    k = np.arange(M)
    nodes  = np.cos(np.pi * (k + 0.5) / M)           # Chebyshev nodes on [-1,1]
    f_vals = _soft_threshold(nodes * R, kappa)        # target function values
    c = np.array([
        (2.0 / M) * np.sum(f_vals * np.cos(np.pi * j * (k + 0.5) / M))
        for j in range(M)
    ])
    c[0] /= 2.0   # standard Chebyshev series convention
    return c


def eval_cheby(w: np.ndarray, coeffs: np.ndarray, R: float) -> np.ndarray:
    """
    Evaluate Chebyshev series at w/R using Clenshaw recurrence.
    coeffs[0] is already c_0/2 (server.cpp convention) → plain sum, no correction.
    Mirrors OpenFHE EvalChebyshevSeries.
    """
    t = np.clip(w / R, -1.0, 1.0)
    d = len(coeffs) - 1
    if d == 0:
        return coeffs[0] * np.ones_like(t)
    T_prev, T_curr = np.ones_like(t), t.copy()
    y = coeffs[0] * T_prev + coeffs[1] * T_curr
    for j in range(2, d + 1):
        T_next = 2.0 * t * T_curr - T_prev
        y += coeffs[j] * T_next
        T_prev, T_curr = T_curr, T_next
    return y


# ============================================================================
# ServerSim
# ============================================================================
class ServerSim:
    """
    Mirrors server.cpp TRIAD logic for plaintext simulation.

    Owns global state: self.z (current z iterate).
    Does NOT hold references to ClientSim objects — all client data is passed
    explicitly as arguments, mirroring the network message interface.
    """

    def __init__(self, A_full: np.ndarray, b_full: np.ndarray):
        self.A = A_full
        self.b = b_full
        self.z = np.zeros(N_FEAT)

    # ------------------------------------------------------------------
    # Phase 1 — Bootstrap R
    # ------------------------------------------------------------------
    def phase1_bootstrap_r(self, x0_list: list):
        """
        Aggregate enc(x_i^(0)) from all clients, compute initial R.
        Mirrors server.cpp Phase 1 EvalInnerProduct + MultipartyDecryptFusion block.

        Returns (currentR, coeffs).
        """
        x_bar = np.mean(x0_list, axis=0)            # (1/K)*sum(x_k^(0))
        sumSq = float(np.dot(x_bar, x_bar))
        currentR = max(
            1.5 * np.sqrt(sumSq / N_FEAT),
            3.0 * KAPPA,
            1.5,
        )
        self.z = np.zeros(N_FEAT)                   # initialise z = 0
        return currentR, cheby_coeffs(currentR)

    # ------------------------------------------------------------------
    # Per-iteration helpers
    # ------------------------------------------------------------------
    def prepare_v(self, u_k: np.ndarray) -> np.ndarray:
        """
        Compute v = z - u_k to send to client k (masked-decrypt simulation).
        Mirrors server.cpp maskedDecryptSend for the v_i = z - u_i channel.
        """
        return self.z - u_k

    def aggregate_w(self, x_list: list, u_list: list) -> np.ndarray:
        """
        Compute w = (1/K) * sum_k(x_k + u_k).
        Mirrors server.cpp step (e): enc(w) = EvalMult(sum(enc(x)+enc(u)), 1/K).
        """
        w = np.zeros(N_FEAT)
        for x, u in zip(x_list, u_list):
            w += x + u
        return w / NUM_PARTIES

    def compute_R_raw(self, w: np.ndarray) -> float:
        """
        CRC: compute Psi = ||w||^2, R_raw = alpha * sqrt(Psi/n).
        Mirrors server.cpp EvalInnerProduct + MultipartyDecryptFusion for Psi.
        """
        Psi = float(np.dot(w, w))
        return ALPHA_CRC * np.sqrt(Psi / N_FEAT)

    def update_R(self, it: int, R_raw: float, currentR: float, safe_all: bool):
        """
        Update R using CRC result and recompute Chebyshev coefficients.
        Mirrors server.cpp CRC R-update block exactly (same if/elif/elif logic).
        """
        if R_raw > currentR:
            currentR = R_raw
        elif not safe_all:
            currentR = currentR / DELTA_SAFE
        elif it > SHRINK_WARM and safe_all:        # strict > mirrors server.cpp
            currentR = max(R_raw, GAMMA_SMOOTH * currentR)
        return currentR, cheby_coeffs(currentR)

    def z_update(self, w: np.ndarray, currentR: float, coeffs: np.ndarray) -> np.ndarray:
        """
        Chebyshev soft-threshold z-update.
        Mirrors server.cpp EvalChebyshevSeries(encWn, coeffs, -1.0, 1.0).
        """
        self.z = eval_cheby(w, coeffs, currentR)
        return self.z.copy()

    # ------------------------------------------------------------------
    # Objective monitoring (side-decrypt equivalent)
    # ------------------------------------------------------------------
    def compute_objective(self, z: np.ndarray) -> float:
        """
        (1/2)||Az-b||^2 + lambda*||z||_1.
        Mirrors server.cpp computeObjective called after MultipartyDecryptFusion.
        """
        r = self.A @ z - self.b
        return float(0.5 * np.dot(r, r) + LAMBDA * np.sum(np.abs(z)))

    @staticmethod
    def check_explosion(obj: float, prev_obj) -> bool:
        """
        Detect numerical explosion (mirrors server.cpp side-decrypt explosion logic).
        Returns True if:
          - obj is NaN or Inf
          - obj > EXPLOSION_ABS
          - obj jumped > EXPLOSION_JUMP × previous value (and obj > EXPLOSION_MIN)
        """
        if np.isnan(obj) or np.isinf(obj):
            return True
        if obj > EXPLOSION_ABS:
            return True
        if (prev_obj is not None and prev_obj > 0
                and obj > prev_obj * EXPLOSION_JUMP
                and obj > EXPLOSION_MIN):
            return True
        return False
