"""
client_sim.py — TRIAD client-side simulation (mirrors client.cpp).

Each ClientSim instance represents one Raspberry Pi party.
Messages between server and client are plain numpy arrays (no encryption).
"""

import numpy as np

# ============================================================================
# Parameters — must match server_sim.py and solve/client.cpp
# ============================================================================
N_FEAT      = 200
NUM_PARTIES = 3
RHO         = 1.0
LAMBDA      = 0.1
KAPPA       = LAMBDA / (RHO * NUM_PARTIES)   # = 0.03333...
DELTA_SAFE  = 0.95


class ClientSim:
    """
    Mirrors client.cpp TRIAD logic for plaintext simulation.

    Precompute (mirrors client.cpp precompute block):
        M_inv = (A_i^T A_i + rho*I)^{-1}   [direct solve; real code uses Woodbury for p=4088]
        g     = A_i^T b_i

    Per-iteration state maintained: self.x (last x-update), self.u (dual variable).
    """

    def __init__(self, party_id: int, A: np.ndarray, b: np.ndarray):
        self.pid = party_id
        n = N_FEAT
        # Precompute inverse system: M_inv = (A^T A + rho I)^{-1}
        self.M_inv = np.linalg.solve(A.T @ A + RHO * np.eye(n), np.eye(n))
        self.g     = A.T @ b          # A_i^T b_i
        self.x     = np.zeros(n)      # x_i  (updated each iteration)
        self.u     = np.zeros(n)      # dual variable u_i

    # ------------------------------------------------------------------
    # Phase 1 — Bootstrap R
    # ------------------------------------------------------------------
    def phase1_send_x0(self) -> np.ndarray:
        """
        Compute x_i^(0) = M_inv @ g  (z=0, u=0 initialization).
        'Encrypt and send enc(x_i^(0))' to server — returns plaintext in simulation.
        Mirrors client.cpp Phase 1 enc(x_i) send.
        """
        self.x = self.M_inv @ self.g
        return self.x.copy()

    # ------------------------------------------------------------------
    # Phase 2 — Per-iteration steps
    # ------------------------------------------------------------------
    def recv_v_do_x_update(self, v: np.ndarray) -> np.ndarray:
        """
        (steps a–c) Receive v = z - u_i from server (masked-decrypt result).
        Compute x_i = M_inv @ (g + rho * v).
        'Encrypt and send enc(x_i)' back to server.
        Mirrors client.cpp x-update block.
        """
        self.x = self.M_inv @ (self.g + RHO * v)
        return self.x.copy()

    def crc_check_safe(self, R_raw: float, currentR: float) -> bool:
        """
        (step e) CRC safety check.
        Receive R_raw from server.
        Check: safe = (max|x_i + u_i| <= delta_safe * currentR).
        Uses currentR (not R_raw) for bound — mirrors the CRC fix in client.cpp.
        Return safe bool to server.
        """
        max_w = float(np.max(np.abs(self.x + self.u)))
        return max_w <= DELTA_SAFE * currentR

    def recv_z_do_u_update(self, z_new: np.ndarray) -> None:
        """
        (steps f–g) Receive z_new from server.
        Update u_i = u_i + x_i - z_new.
        'Encrypt and send enc(u_i)' back to server.
        Mirrors client.cpp u-update and send.
        """
        self.u = self.u + self.x - z_new
