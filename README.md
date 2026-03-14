# Distributed Lasso via Multi-Key Homomorphic Encryption (TRIAD)

A privacy-preserving distributed LASSO solver using threshold CKKS homomorphic encryption, implemented with [OpenFHE](https://github.com/openfheorg/openfhe-development) v1.5.0.

## Overview

**TRIAD** (Threshold-based Range-adaptive Iterative ADMM under encryption) solves the federated LASSO problem:

$$\min_{z} \frac{1}{2}\|Az - b\|^2 + \lambda \|z\|_1$$

where the data matrix $A$ and labels $b$ are horizontally partitioned across $K$ clients. The server never sees plaintext client data; all x-updates happen under CKKS encryption with multiparty decryption.

### Key Features

- **Threshold CKKS**: 3-party joint key generation; no single party can decrypt alone
- **Adaptive Chebyshev soft-threshold (CRC)**: range parameter $R$ adapts online via a Chebyshev Range Check, avoiding numerical explosion
- **Woodbury identity**: client-side x-update uses $(m_i \times m_i)$ matrix inversion instead of $(p \times p)$, enabling large feature dimensions
- **Bootstrap-refresh**: ciphertext level management with full multiparty re-encryption when levels drop below threshold

---

## Repository Structure

```
.
├── keygen.cpp            # Key generation for synthetic experiment
├── server.cpp            # TRIAD server — synthetic dataset
├── client.cpp            # TRIAD client — synthetic dataset
├── plot_convergence.py   # Convergence plot (PlainADMM / Static-R / TRIAD)
├── plot_time.py          # Per-iteration latency breakdown plot
├── sparsity_analysis.py  # Sparsity sensitivity analysis script
├── cMakelists.txt        # CMake build (server/client, Windows)
├── CMakeLists_pi.txt     # CMake build (client, Raspberry Pi)
│
└── solve_real/           # Riboflavin real-data experiment (71×4088)
    ├── keygen_real.cpp       # Key generation (BatchSize=8192 for N=4088)
    ├── server_real.cpp       # TRIAD server — Riboflavin dataset
    ├── client_real.cpp       # TRIAD client — Riboflavin dataset
    ├── plot_convergence_real.py  # Convergence plot for real-data experiment
    └── CMakeLists.txt        # CMake build (-Wa,-mbig-obj for MinGW)
```

---

## System Architecture

```
[Windows Server]
       |
  +---------+---------+
  |         |         |
[Pi 0]   [Pi 1]   [Pi 2]
party_0  party_1  party_2
port 10000 10001  10002
```

- Server: Windows x86-64, runs key aggregation + ADMM z-update (Chebyshev)
- Clients: Raspberry Pi (ARM), each holds a private data partition + secret key share

---

## Dependencies

| Component | Version |
|-----------|---------|
| OpenFHE   | v1.5.0  |
| CMake     | ≥ 3.16  |
| Compiler  | MinGW-w64 (Windows) / GCC (Pi) |
| Python    | ≥ 3.8 (for plotting) |

Python packages: `numpy pandas matplotlib`

---

## Experiment 1 — Synthetic Dataset (`solve/`)

**Problem**: $n=100$ samples, $p=200$ features, 3 clients (rows split evenly), ground-truth $x^*$ with 10% sparsity. $\lambda = 0.1$.

### Build (Windows server)

```bash
mkdir build && cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
mingw32-make -j4
```

### Build (Raspberry Pi clients)

```bash
mkdir build && cd build
cmake .. -f ../CMakeLists_pi.txt -DCMAKE_BUILD_TYPE=Release
make -j2
```

### Key Generation

```bash
# On server
cd build && ./keygen.exe
# Distribute keys/ to all 3 Pis:
scp -r keys/ huang@<Pi0_IP>:~/client/build/
scp -r keys/ huang@<Pi1_IP>:~/client/build/
scp -r keys/ huang@<Pi2_IP>:~/client/build/
```

### Run

```bash
# Start clients first (one terminal per Pi)
ssh huang@<Pi0_IP> "cd ~/client/build && ./client 0"
ssh huang@<Pi1_IP> "cd ~/client/build && ./client 1"
ssh huang@<Pi2_IP> "cd ~/client/build && ./client 2"

# Once all clients show "Listening...", start server
cd build && ./server.exe
```

Optional flag `--triad-only` skips the PlainADMM baseline.

### Plot

```bash
python plot_convergence.py   # reads build/*.csv, outputs figures/
```

---

## Experiment 2 — Riboflavin Real Dataset (`solve_real/`)

**Problem**: Riboflavin production dataset, $n=71$ samples, $p=4088$ genes, 3 clients (rows 0–23, 24–47, 48–70). $\lambda = 0.01$.

Data preprocessing: column-wise L2 normalization on $X$, mean-centering on $y$ (applied identically on server and all clients).

### Build (Windows server)

```bash
cd solve_real
mkdir build && cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
mingw32-make -j4
```

### Build (Raspberry Pi clients)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2
```

### Data Setup

Place in `build/data/`:
```
data/
  riboflavin_X.csv   # 71 × 4088, header row, comma-separated
  riboflavin_y.csv   # 71 × 1,    header row, comma-separated
```

The Riboflavin dataset is from the R package `hdi` (`riboflavin`). To export:

```R
library(hdi)
data(riboflavin)
write.csv(t(riboflavin$x), "riboflavin_X.csv")
write.csv(riboflavin$y,    "riboflavin_y.csv")
```

### Key Generation

BatchSize must be ≥ 4088; `keygen_real` uses BatchSize=8192.

```bash
cd build && ./keygen_real.exe
scp -r keys/ huang@<Pi0_IP>:~/client_real/build/
scp -r keys/ huang@<Pi1_IP>:~/client_real/build/
scp -r keys/ huang@<Pi2_IP>:~/client_real/build/
```

### Run

```bash
# Start clients first
ssh huang@<Pi0_IP> "cd ~/client_real/build && pkill -f client_real; ./client_real 0"
ssh huang@<Pi1_IP> "cd ~/client_real/build && pkill -f client_real; ./client_real 1"
ssh huang@<Pi2_IP> "cd ~/client_real/build && pkill -f client_real; ./client_real 2"

# Once all show "Listening...", start server
cd solve_real/build && ./server_real.exe
```

Output logs (in `build/`):
- `plaintext_admm_real_log.csv` — PlainADMM baseline (100 iterations)
- `Adaptive_TRIAD_real_log.csv` — TRIAD with CRC (100 iterations)
- `timing_real_log.csv` — per-step latency breakdown

### Plot

```bash
cd solve_real && python plot_convergence_real.py
# Output: solve_real/figures/triad_real_convergence.{pdf,png}
```

---

## Protocol Details

### Phase 0 — Threshold Key Setup
Multiparty CKKS key generation (OpenFHE threshold API):
1. Party 0 generates initial key pair
2. Parties 1 & 2 contribute chained key pairs
3. Joint public key + eval mult/rot/sum keys are assembled and broadcast

### Phase 1 — Bootstrap R
Each client encrypts $x_i^{(0)} = (A_i^\top A_i + \rho I)^{-1} A_i^\top b_i$ (computed via Woodbury identity for large $p$).
Server aggregates → computes $\|\bar{x}^{(0)}\|_2$ via multiparty inner product decryption → sets initial $R$.

### Phase 2 — ADMM Iterations with CRC
Each iteration:
1. Server sends masked $\text{enc}(z - u_i)$ for client x-update
2. Client returns $\text{enc}(x_i)$
3. Server computes $\text{enc}(w) = \frac{1}{K}\sum_i \text{enc}(x_i + u_i)$
4. **CRC** (every 5 iters after warmup): decrypt $\|w\|^2$ → update $R$ adaptively
5. Server applies Chebyshev soft-threshold: $z \leftarrow S_\kappa^{(R)}(w)$
6. Refresh if remaining levels < threshold
7. Side-decrypt for objective monitoring

### Phase 3 — Final Result
Multiparty decryption of final $\text{enc}(z)$.

---

## Notes

- Secret key shares are **never** sent to the server; multiparty decryption uses partial decryption shares only
- The CRC prevents Chebyshev approximation failure by keeping $w/R \in [-1, 1]$
- `--triad-only` flag skips PlainADMM baseline (useful when re-running only TRIAD)
