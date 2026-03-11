# TRIAD Deployment Guide

## Architecture

```
[Main Machine - Server]
        |
   +---------+---------+
   |         |         |
[Pi 0]    [Pi 1]    [Pi 2]
party_0   party_1   party_2
port 9000 port 9001 port 9002
```

## Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

On Raspberry Pi (cross-compile or build on Pi directly):
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4   # use -j2 if Pi runs out of memory during compile
```

## Run

### 1. Start server first (on main machine)
```bash
./server
```
Server will block waiting for all 3 clients to connect.

### 2. Start each client (on each Raspberry Pi)
```bash
# Pi 0
./client 0 <server_ip>

# Pi 1
./client 1 <server_ip>

# Pi 2
./client 2 <server_ip>
```

Start all 3 clients within a few seconds of each other.

## Protocol Phases

### Phase 0 - Key Setup
1. Server generates party-0 keypair
2. Broadcasts party-0 public key to all clients
3. Client 1 and 2 generate chained keys, send public+secret keys to server
4. Server builds joint public key + eval keys
5. Broadcasts joint public key to all clients

### Phase 1 - Initial R Bootstrap
1. Each client computes x_i^(0) = (A_i^T A_i + rhoI)^{-1} A_i^T b_i
2. Encrypts with joint public key, sends enc(x_i^(0)) to server
3. Server aggregates, computes ||x^(0)||_2 via CRC, sets R = 1.5 * ||x^(0)||_2
4. Broadcasts initial R to all clients

### Phase 2 - ADMM-CRC Iterations (50 rounds)
Each iteration:
1. Server computes enc(z-u), sends to all clients
2. Clients partially decrypt z-u, server fuses -> plaintext z-u
3. Each client computes x_i update locally
4. Each client encrypts x_i, sends enc(x_i) to server
5. Server aggregates enc(x), computes enc(w) = enc(x) + enc(u)
6. Every 5 iters (after warmup): CRC computes ||w||_2 and ||w||_4,
   adaptively shrinks R if safe
7. Server runs EvalChebyshev on enc(w) -> enc(z)
8. Server refreshes if needed (level < threshold)
9. Server updates enc(u) = enc(u) + enc(x) - enc(z)
10. Server broadcasts updated R + iter-done signal

## Output Files (server-side)
- `triad_convergence.csv`: iteration, objective, R history, CRC/refresh events

## Notes
- Party 0 key is generated server-side in simulation mode.
  For fully distributed deployment, remove server-side party-0 key gen
  and have Pi 0 send its public key first.
- The server holds all secret key shares in simulation mode (honest server).
  For production, use threshold decryption so no single party holds all shares.
- Partial decrypt protocol: each client sends its partial to server,
  server fuses. This is implemented in client.cpp Step 1.