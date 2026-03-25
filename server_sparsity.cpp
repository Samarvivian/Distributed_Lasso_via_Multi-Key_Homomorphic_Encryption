/**
 * @file server_sparsity.cpp
 * @brief TRIAD Sparsity Sensitivity Experiment — Server (Windows)
 *
 * Runs 5 encrypted TRIAD experiments (sparsity 10%–50%) sequentially.
 * Synthetic data: A is 150×200 Gaussian (seed=42, column-normalized).
 * For each sparsity level: generates x_true → b=Ax_true → distributes b_i
 * to clients → runs full threshold-CKKS TRIAD for maxIter=50 rounds.
 * Explosion detection: if objective NaN/Inf or > 1e6, sends MAGIC_ABORT_EXP
 * to all clients and skips to the next sparsity level.
 *
 * Usage:
 *   ./server_sparsity.exe
 *
 * Requires:
 *   keys/  directory (from keygen.exe with N_FEAT=200, BatchSize=256)
 *
 * Output logs (in working directory):
 *   sparsity_10_log.csv  ...  sparsity_50_log.csv
 *   columns: iter,R,remLev,crc,refresh,objective,elapsed_s
 */

#include "openfhe/pke/openfhe.h"
#include "openfhe/pke/scheme/ckksrns/ckksrns-ser.h"
#include "openfhe/pke/key/key-ser.h"
#include "openfhe/pke/cryptocontext-ser.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <iomanip>
#include <numeric>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
  static void netInit() { WSADATA w; WSAStartup(MAKEWORD(2,2), &w); }
  static void netClose(int fd) { closesocket((SOCKET)(uintptr_t)fd); }
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  static void netInit() {}
  static void netClose(int fd) { close(fd); }
#endif

using namespace lbcrypto;
using namespace std;

// ============================================================================
// Config — must match client_sparsity.cpp exactly
// ============================================================================
static const int    NUM_PARTIES    = 3;
static const int    BASE_PORT      = 10200;
static const size_t N_FEAT         = 200;
static const int    M_ROWS         = 50;      // rows per party
static const int    TOTAL_ROWS     = M_ROWS * NUM_PARTIES;  // 150
static const double rho            = 1.0;
static const double lambda_lasso   = 0.1;
static const int    maxIter        = 50;
static const int    updateInterval = 5;
static const int    shrinkWarmup   = 5;
static const int    chebyDegree    = 15;
static const double alpha_crc      = 4.0;
static const double gamma_smooth   = 0.8;
static const double delta_safe     = 0.95;
static const int    Lmin_levels    = 6;
static const double kappa          = lambda_lasso / (rho * NUM_PARTIES);

static const double SPARSITY_LEVELS[] = {0.30, 0.40};
static const int    N_SPARSITY        = 2;

// Explosion detection thresholds
static const double EXPLOSION_ABS  = 1e6;
static const double EXPLOSION_JUMP = 100.0;
static const double EXPLOSION_MIN  = 100.0;

static const char* CLIENT_IPS[3] = {
    "192.168.186.223",
    "192.168.186.33",
    "192.168.186.214"
};

// Magic constants — must match client_sparsity.cpp
static const uint32_t MAGIC_READY         = 0xCAFEBABE;
static const uint32_t MAGIC_REFRESH       = 0xABCD1234;
static const uint32_t MAGIC_ITERDONE      = 0x00000001;
static const uint32_t MAGIC_END           = 0xFFFFFFFF;
static const uint32_t MAGIC_ALL_DONE      = 0xEEEEEEEE;
static const uint32_t MAGIC_SPARSITY_EXP  = 0xDD000001;
static const uint32_t MAGIC_CONTINUE_ITER = 0x00000003;
static const uint32_t MAGIC_ABORT_EXP     = 0xDD000002;

// ============================================================================
// Network helpers (identical to server_real.cpp)
// ============================================================================
static void sendAll(int fd, const char* buf, size_t len) {
    size_t s = 0;
    while (s < len) {
        ssize_t r = send(fd, buf + s, (int)(len - s), 0);
        if (r <= 0) throw runtime_error("send failed");
        s += r;
    }
}
static void recvAll(int fd, char* buf, size_t len) {
    size_t g = 0;
    while (g < len) {
        ssize_t r = recv(fd, buf + g, (int)(len - g), 0);
        if (r <= 0) throw runtime_error("recv failed");
        g += r;
    }
}
template<typename T>
static void sendObj(int fd, const T& obj, const CryptoContext<DCRTPoly>& cc) {
    ostringstream oss;
    Serial::Serialize(obj, oss, SerType::BINARY);
    string d = oss.str();
    uint32_t n = htonl((uint32_t)d.size());
    sendAll(fd, (char*)&n, 4);
    sendAll(fd, d.data(), d.size());
}
template<typename T>
static T recvObj(int fd, const CryptoContext<DCRTPoly>& cc) {
    uint32_t n;
    recvAll(fd, (char*)&n, 4);
    string d(ntohl(n), '\0');
    recvAll(fd, d.data(), d.size());
    istringstream iss(d);
    T obj;
    Serial::Deserialize(obj, iss, SerType::BINARY);
    return obj;
}
static void sendVec(int fd, const vector<double>& v) {
    uint32_t n = htonl((uint32_t)(v.size() * 8));
    sendAll(fd, (char*)&n, 4);
    sendAll(fd, (char*)v.data(), v.size() * 8);
}
static vector<double> recvVec(int fd) {
    uint32_t n; recvAll(fd, (char*)&n, 4);
    uint32_t bytes = ntohl(n);
    vector<double> v(bytes / 8);
    recvAll(fd, (char*)v.data(), bytes);
    return v;
}
static void     sendD  (int fd, double   v) { sendAll(fd, (char*)&v, 8); }
static void     sendU32(int fd, uint32_t v) { uint32_t n = htonl(v); sendAll(fd, (char*)&n, 4); }
static uint32_t recvU32(int fd)             { uint32_t n; recvAll(fd, (char*)&n, 4); return ntohl(n); }
static bool     recvBool(int fd)            { uint8_t b; recvAll(fd, (char*)&b, 1); return b != 0; }

template<typename T>
static void bcast(const vector<int>& fds, const T& obj, const CryptoContext<DCRTPoly>& cc) {
    for (int fd : fds) sendObj(fd, obj, cc);
}
static void bcastU32(const vector<int>& fds, uint32_t v) { for (int fd : fds) sendU32(fd, v); }
static void bcastD  (const vector<int>& fds, double   v) { for (int fd : fds) sendD  (fd, v); }

// ============================================================================
// Chebyshev soft-threshold (identical to server_real.cpp)
// ============================================================================
static vector<double> chebyCoeffs(double R, int deg) {
    int M = deg + 1;
    vector<double> c(M, 0.0);
    for (int j = 0; j < M; j++) {
        double s = 0;
        for (int k = 0; k < M; k++) {
            double nd = cos(M_PI * (k + 0.5) / M);
            double x  = nd * R;
            double fx = (x > kappa) ? x - kappa : (x < -kappa) ? x + kappa : 0.0;
            s += fx * cos(M_PI * j * (k + 0.5) / M);
        }
        c[j] = (2.0 / M) * s;
    }
    c[0] /= 2.0;
    return c;
}

// ============================================================================
// maskedDecryptSend (identical to server_real.cpp)
// ============================================================================
static void maskedDecryptSend(
    int targetId,
    const vector<int>& fds,
    const Ciphertext<DCRTPoly>& enc_val,
    const Ciphertext<DCRTPoly>& enc_r,
    const CryptoContext<DCRTPoly>& cc,
    size_t vecLen)
{
    auto enc_masked = cc->EvalAdd(enc_val, enc_r);
    bcast(fds, enc_masked, cc);
    vector<Ciphertext<DCRTPoly>> shares(NUM_PARTIES);
    for (int i = 0; i < NUM_PARTIES; i++)
        shares[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
    Plaintext pt;
    cc->MultipartyDecryptFusion(shares, &pt);
    pt->SetLength(vecLen);
    sendVec(fds[targetId], pt->GetRealPackedValue());
}

// ============================================================================
// Math helpers
// ============================================================================
using Mat = vector<vector<double>>;

static vector<double> vecadd(const vector<double>& a, const vector<double>& b) {
    vector<double> c(a.size()); for (size_t i=0;i<a.size();i++) c[i]=a[i]+b[i]; return c;
}

static double computeObjective(const vector<double>& z,
                                const Mat& A, const vector<double>& b) {
    double data = 0.0;
    for (size_t i = 0; i < A.size(); i++) {
        double Az = 0.0;
        for (size_t j = 0; j < z.size(); j++) Az += A[i][j] * z[j];
        double d = Az - b[i]; data += d * d;
    }
    data *= 0.5;
    double l1 = 0.0;
    for (double v : z) l1 += fabs(v);
    return data + lambda_lasso * l1;
}

static bool checkExplosion(double obj, double prev_obj) {
    if (std::isnan(obj) || std::isinf(obj)) return true;
    if (obj > EXPLOSION_ABS) return true;
    if (prev_obj > 0 && obj > prev_obj * EXPLOSION_JUMP && obj > EXPLOSION_MIN) return true;
    return false;
}

static double computeMSE(const vector<double>& z, const vector<double>& x_true) {
    double s = 0.0;
    for (size_t i = 0; i < z.size(); i++) {
        double d = z[i] - x_true[i]; s += d * d;
    }
    return s / z.size();
}

// ============================================================================
// Data generation
// ============================================================================
// Generate full A matrix (150×200, column-normalized Gaussian, seed=42)
static void generateA(Mat& A_full, vector<Mat>& A_parties) {
    mt19937_64 rng(42);
    normal_distribution<double> ndist(0.0, 1.0);

    A_full.assign(TOTAL_ROWS, vector<double>(N_FEAT));
    for (int i = 0; i < TOTAL_ROWS; i++)
        for (size_t j = 0; j < N_FEAT; j++)
            A_full[i][j] = ndist(rng);

    // Column-normalize (L2)
    for (size_t j = 0; j < N_FEAT; j++) {
        double nm = 0;
        for (int i = 0; i < TOTAL_ROWS; i++) nm += A_full[i][j] * A_full[i][j];
        nm = sqrt(nm);
        if (nm > 1e-12)
            for (int i = 0; i < TOTAL_ROWS; i++) A_full[i][j] /= nm;
    }

    // Split into partitions
    A_parties.resize(NUM_PARTIES);
    for (int p = 0; p < NUM_PARTIES; p++) {
        A_parties[p].clear();
        for (int i = p * M_ROWS; i < (p + 1) * M_ROWS; i++)
            A_parties[p].push_back(A_full[i]);
    }
}

// Generate sparse x_true and compute b = A*x_true
static void generateGroundTruth(const Mat& A_full, double sparsity, int sp_idx,
                                  vector<double>& x_true, vector<double>& b_full) {
    int n_nz = max(1, (int)round(sparsity * (double)N_FEAT));
    mt19937_64 rng_x(200 + sp_idx * 1000);
    uniform_real_distribution<double> ampDist(0.5, 2.0);

    x_true.assign(N_FEAT, 0.0);
    // Reservoir sampling for non-zero positions
    vector<size_t> perm(N_FEAT);
    iota(perm.begin(), perm.end(), 0);
    for (int i = 0; i < n_nz; i++) {
        size_t j = i + rng_x() % (N_FEAT - i);
        swap(perm[i], perm[j]);
    }
    for (int i = 0; i < n_nz; i++) {
        double amp = ampDist(rng_x);
        x_true[perm[i]] = (rng_x() % 2 == 0) ? amp : -amp;
    }

    // b = A * x_true
    b_full.assign(TOTAL_ROWS, 0.0);
    for (int i = 0; i < TOTAL_ROWS; i++)
        for (size_t j = 0; j < N_FEAT; j++)
            b_full[i] += A_full[i][j] * x_true[j];
}

// ============================================================================
// Main
// ============================================================================
int main() {
    auto t0 = chrono::high_resolution_clock::now();
    auto elapsed = [&]() {
        return chrono::duration<double>(chrono::high_resolution_clock::now() - t0).count();
    };

    netInit();
    cout << "=== TRIAD Sparsity Server (K=" << NUM_PARTIES << ") ===" << endl;
    cout << "  N_FEAT=" << N_FEAT << " M_ROWS=" << M_ROWS
         << " lambda=" << lambda_lasso << " kappa=" << kappa
         << " maxIter=" << maxIter << " Lmin=" << Lmin_levels << endl;

    // -----------------------------------------------------------------------
    // Generate data
    // -----------------------------------------------------------------------
    cout << "\n[Data] Generating A (" << TOTAL_ROWS << "x" << N_FEAT
         << ", seed=42, column-normalized)..." << endl;
    Mat A_full;
    vector<Mat> A_parties;
    generateA(A_full, A_parties);
    cout << "  A generated (t=" << elapsed() << "s)" << endl;

    // -----------------------------------------------------------------------
    // Load keys
    // -----------------------------------------------------------------------
    cout << "\n[Phase 0] Loading keys..." << endl;
    CryptoContext<DCRTPoly> cc;
    Serial::DeserializeFromFile("keys/cryptocontext.bin", cc, SerType::BINARY);
    cc->Enable(PKE); cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE); cc->Enable(ADVANCEDSHE); cc->Enable(MULTIPARTY);

    PublicKey<DCRTPoly> jointPK;
    Serial::DeserializeFromFile("keys/joint_pk.bin", jointPK, SerType::BINARY);

    vector<EvalKey<DCRTPoly>> emVec;
    Serial::DeserializeFromFile("keys/eval_mult_key.bin", emVec, SerType::BINARY);
    cc->InsertEvalMultKey(emVec);

    map<usint, EvalKey<DCRTPoly>> rotMap;
    Serial::DeserializeFromFile("keys/eval_rot_key.bin", rotMap, SerType::BINARY);
    cc->InsertEvalAutomorphismKey(make_shared<map<usint,EvalKey<DCRTPoly>>>(rotMap));

    map<usint, EvalKey<DCRTPoly>> sumMap;
    Serial::DeserializeFromFile("keys/eval_sum_key.bin", sumMap, SerType::BINARY);
    cc->InsertEvalSumKey(make_shared<map<usint,EvalKey<DCRTPoly>>>(sumMap));

    size_t totalLevels = cc->GetCryptoParameters()
                           ->GetElementParams()->GetParams().size();
    cout << "  Keys loaded. RingDim=" << cc->GetRingDimension()
         << " BatchSize=" << cc->GetEncodingParams()->GetBatchSize()
         << " totalLevels=" << totalLevels
         << " (t=" << elapsed() << "s)" << endl;

    // -----------------------------------------------------------------------
    // Connect to clients
    // -----------------------------------------------------------------------
    cout << "\n[Connect] Connecting to clients..." << endl;
    vector<int> fds(NUM_PARTIES, -1);
    for (int i = 0; i < NUM_PARTIES; i++) {
        for (int attempt = 0; attempt < 120; attempt++) {
            int s = socket(AF_INET, SOCK_STREAM, 0);
            sockaddr_in a{}; a.sin_family = AF_INET;
            a.sin_port = htons(BASE_PORT + i);
            inet_pton(AF_INET, CLIENT_IPS[i], &a.sin_addr);
            if (connect(s, (sockaddr*)&a, sizeof(a)) == 0) { fds[i] = s; break; }
            netClose(s);
#ifdef _WIN32
            Sleep(1000);
#else
            sleep(1);
#endif
        }
        if (fds[i] < 0) throw runtime_error("Cannot connect to client " + to_string(i));
        cout << "  Client " << i << " connected (t=" << elapsed() << "s)" << endl;
    }
    for (int i = 0; i < NUM_PARTIES; i++) {
        if (recvU32(fds[i]) != MAGIC_READY)
            throw runtime_error("Client " + to_string(i) + " not ready");
    }
    cout << "  All clients ready (t=" << elapsed() << "s)" << endl;

    // -----------------------------------------------------------------------
    // Distribute A_i to each client (one-time, before experiment loop)
    // -----------------------------------------------------------------------
    cout << "\n[Data] Distributing A_i to clients..." << endl;
    for (int p = 0; p < NUM_PARTIES; p++) {
        // Send dimensions + row-by-row data
        uint32_t nrows = (uint32_t)A_parties[p].size();
        uint32_t ncols = (uint32_t)N_FEAT;
        sendU32(fds[p], nrows);
        sendU32(fds[p], ncols);
        for (auto& row : A_parties[p]) sendVec(fds[p], row);
    }
    // Wait for clients to confirm data received and precomputed
    for (int i = 0; i < NUM_PARTIES; i++) {
        if (recvU32(fds[i]) != MAGIC_READY)
            throw runtime_error("Client " + to_string(i) + " not ready after data receive");
    }
    cout << "  All clients confirmed A_i received and M_inv precomputed (t="
         << elapsed() << "s)" << endl;

    // -----------------------------------------------------------------------
    // Experiment loop over sparsity levels
    // -----------------------------------------------------------------------
    for (int sp_idx = 0; sp_idx < N_SPARSITY; sp_idx++) {
        double sparsity = SPARSITY_LEVELS[sp_idx];
        int    pct      = (int)(sparsity * 100);
        int    n_nz     = max(1, (int)round(sparsity * (double)N_FEAT));

        cout << "\n===== Sparsity " << pct << "% (" << n_nz << "/" << N_FEAT
             << " non-zeros) =====" << endl;

        // Generate ground-truth x_true, b_full = A*x_true
        vector<double> x_true, b_full;
        generateGroundTruth(A_full, sparsity, sp_idx, x_true, b_full);
        cout << "  x_true generated, ||x*||_1="
             << fixed << setprecision(4)
             << [&](){ double s=0; for (double v:x_true) s+=fabs(v); return s; }()
             << endl;

        // Signal new experiment + send b_i to each client
        bcastU32(fds, MAGIC_SPARSITY_EXP);
        bcastD(fds, sparsity);
        for (int p = 0; p < NUM_PARTIES; p++) {
            vector<double> b_i(b_full.begin() + p * M_ROWS,
                               b_full.begin() + (p + 1) * M_ROWS);
            sendVec(fds[p], b_i);
        }
        // Wait for clients to confirm ready for this experiment
        for (int i = 0; i < NUM_PARTIES; i++) {
            if (recvU32(fds[i]) != MAGIC_READY)
                throw runtime_error("Client not ready for sparsity experiment");
        }

        // Initialize encrypted state (z=0, u_i=0 for all clients)
        auto ptZero = cc->MakeCKKSPackedPlaintext(vector<double>(N_FEAT, 0.0));
        auto encZ   = cc->Encrypt(jointPK, ptZero);
        vector<Ciphertext<DCRTPoly>> encU(NUM_PARTIES);
        for (int i = 0; i < NUM_PARTIES; i++)
            encU[i] = cc->Encrypt(jointPK, ptZero);

        // ---- Phase 1: Bootstrap R ----
        cout << "  [Phase 1] Bootstrap R..." << endl;
        vector<Ciphertext<DCRTPoly>> encX0(NUM_PARTIES);
        for (int i = 0; i < NUM_PARTIES; i++)
            encX0[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);

        auto encXbar = encX0[0];
        for (int i = 1; i < NUM_PARTIES; i++)
            encXbar = cc->EvalAdd(encXbar, encX0[i]);
        encXbar = cc->EvalMult(encXbar, 1.0 / NUM_PARTIES);

        auto ct_sumSq = cc->EvalInnerProduct(encXbar, encXbar, N_FEAT);
        bcast(fds, ct_sumSq, cc);

        vector<Ciphertext<DCRTPoly>> sh0(NUM_PARTIES);
        for (int i = 0; i < NUM_PARTIES; i++)
            sh0[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
        Plaintext ptSq;
        cc->MultipartyDecryptFusion(sh0, &ptSq);
        ptSq->SetLength(1);
        double sumSq    = max(0.0, ptSq->GetRealPackedValue()[0]);
        double currentR = max({1.5 * sqrt(sumSq / N_FEAT), 3.0 * kappa, 1.5});
        bcastD(fds, currentR);
        cout << "    R^(0)=" << fixed << setprecision(4) << currentR << endl;

        auto coeffs = chebyCoeffs(currentR, chebyDegree);

        // ---- Phase 2: ADMM iterations ----
        cout << "  [Phase 2] Running " << maxIter << " iterations..." << endl;

        string log_fname = "sparsity_" + to_string(pct) + "_log.csv";
        ofstream log(log_fname);
        log << "iter,R,remLev,crc,refresh,objective,mse,elapsed_s\n";

        bool exploded  = false;
        bool abort_next = false;
        double prev_obj = -1.0;
        int safeStreak  = 0;

        for (int iter = 0; iter < maxIter; iter++) {
            // Pre-iteration signal: CONTINUE or ABORT
            if (abort_next) {
                cout << "    [ABORT] Sparsity " << pct << "% exploded at iter="
                     << iter << endl;
                bcastU32(fds, MAGIC_ABORT_EXP);
                break;
            }
            bcastU32(fds, MAGIC_CONTINUE_ITER);

            bool did_crc = false, did_ref = false;
            size_t remLev = totalLevels - encZ->GetLevel() - 1;
            cout << "    iter=" << setw(2) << iter
                 << "  R=" << fixed << setprecision(4) << currentR
                 << "  remLev=" << remLev
                 << "  t=" << fixed << setprecision(1) << elapsed() << "s" << endl;
            cout.flush();

            // (a) recv masks from all clients
            vector<Ciphertext<DCRTPoly>> encRv(NUM_PARTIES), encRu(NUM_PARTIES);
            for (int i = 0; i < NUM_PARTIES; i++) {
                encRv[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
                encRu[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
            }

            // (b) masked-decrypt v_i = z - u_i for each client
            for (int i = 0; i < NUM_PARTIES; i++)
                maskedDecryptSend(i, fds, cc->EvalSub(encZ, encU[i]), encRv[i], cc, N_FEAT);

            // (c) collect enc(x_i)
            vector<Ciphertext<DCRTPoly>> encX(NUM_PARTIES);
            for (int i = 0; i < NUM_PARTIES; i++)
                encX[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);

            // (d) masked-decrypt u_i for each client (for CRC safe-check)
            for (int i = 0; i < NUM_PARTIES; i++)
                maskedDecryptSend(i, fds, encU[i], encRu[i], cc, N_FEAT);

            // (e) enc(w) = (1/K)*sum(enc(x_i)+enc(u_i))
            auto encW = cc->EvalAdd(encX[0], encU[0]);
            for (int i = 1; i < NUM_PARTIES; i++)
                encW = cc->EvalAdd(encW, cc->EvalAdd(encX[i], encU[i]));
            encW = cc->EvalMult(encW, 1.0 / NUM_PARTIES);

            // (f) CRC
            if (iter >= shrinkWarmup && iter % updateInterval == 0) {
                did_crc = true;
                auto ct_Psi = cc->EvalInnerProduct(encW, encW, N_FEAT);
                bcast(fds, ct_Psi, cc);
                vector<Ciphertext<DCRTPoly>> psh(NUM_PARTIES);
                for (int i = 0; i < NUM_PARTIES; i++)
                    psh[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
                Plaintext ptP; cc->MultipartyDecryptFusion(psh, &ptP); ptP->SetLength(1);
                double Psi   = max(0.0, ptP->GetRealPackedValue()[0]);
                double R_raw = alpha_crc * sqrt(Psi / N_FEAT);
                bcastD(fds, R_raw);
                bool safeAll = true;
                for (int i = 0; i < NUM_PARTIES; i++) safeAll &= recvBool(fds[i]);
                double oldR = currentR;
                if (R_raw > currentR) {
                    currentR = R_raw;
                    safeStreak = 0;
                } else if (!safeAll) {
                    currentR = currentR / delta_safe;
                    safeStreak = 0;
                } else if (iter > shrinkWarmup && safeAll) {
                    safeStreak++;
                    if (safeStreak >= 3)
                        currentR = max(R_raw, gamma_smooth * currentR);
                }
                coeffs = chebyCoeffs(currentR, chebyDegree);
                cout << "      CRC R_raw=" << R_raw << " R: " << oldR
                     << " -> " << currentR << " safe=" << safeAll
                     << " streak=" << safeStreak << endl;
            } else {
                // Non-CRC: per-iteration safe check (zero extra communication cost)
                bool safeAll = true;
                for (int i = 0; i < NUM_PARTIES; i++) safeAll &= recvBool(fds[i]);
                if (!safeAll) {
                    double oldR = currentR;
                    currentR = currentR / delta_safe;
                    coeffs = chebyCoeffs(currentR, chebyDegree);
                    safeStreak = 0;
                    cout << "      [safe] unsafe, R: " << oldR
                         << " -> " << currentR << endl;
                }
            }

            // (g) Chebyshev z-update
            auto encZnew = cc->EvalChebyshevSeries(
                               cc->EvalMult(encW, 1.0 / currentR), coeffs, -1.0, 1.0);

            // (h) broadcast enc(z), collect new enc(u_i)
            bcast(fds, encZnew, cc);
            for (int i = 0; i < NUM_PARTIES; i++)
                encU[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
            encZ = encZnew;

            // (i) refresh if needed
            remLev = totalLevels - encZ->GetLevel() - 1;
            if ((int)remLev < Lmin_levels) {
                did_ref = true;
                cout << "      [Refresh] remLev=" << remLev << endl;
                bcastU32(fds, MAGIC_REFRESH);
                bcast(fds, encZ, cc);
                vector<Ciphertext<DCRTPoly>> zsh(NUM_PARTIES);
                for (int i = 0; i < NUM_PARTIES; i++)
                    zsh[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
                Plaintext ptZp; cc->MultipartyDecryptFusion(zsh, &ptZp); ptZp->SetLength(N_FEAT);
                encZ = cc->Encrypt(jointPK,
                           cc->MakeCKKSPackedPlaintext(ptZp->GetRealPackedValue()));
                bcast(fds, encZ, cc);
                for (int i = 0; i < NUM_PARTIES; i++)
                    encU[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
                cout << "      [Refresh] done remLev=" << (totalLevels-encZ->GetLevel()-1) << endl;
            } else {
                bcastU32(fds, MAGIC_ITERDONE);
            }

            // (j) side-decrypt for objective monitoring
            double obj = -1.0;
            double mse = -1.0;
            {
                bcast(fds, encZ, cc);
                vector<Ciphertext<DCRTPoly>> objSh(NUM_PARTIES);
                for (int i = 0; i < NUM_PARTIES; i++)
                    objSh[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
                try {
                    Plaintext ptObj; cc->MultipartyDecryptFusion(objSh, &ptObj);
                    ptObj->SetLength(N_FEAT);
                    auto zvec = ptObj->GetRealPackedValue(); zvec.resize(N_FEAT);
                    obj = computeObjective(zvec, A_full, b_full);
                    mse = computeMSE(zvec, x_true);
                    cout << "      obj=" << fixed << setprecision(6) << obj
                         << "  mse=" << setprecision(6) << mse << endl;
                    if (checkExplosion(obj, prev_obj)) {
                        cout << "      [EXPLOSION] obj=" << obj
                             << " prev=" << prev_obj << endl;
                        exploded   = true;
                        abort_next = true;
                    } else {
                        prev_obj = obj;
                    }
                } catch (...) {
                    cout << "      [DECRYPT_FAIL] iter=" << iter << endl;
                    abort_next = true; exploded = true;
                }
            }

            log << iter << "," << currentR << ","
                << (totalLevels - encZ->GetLevel() - 1) << ","
                << did_crc << "," << did_ref << ","
                << (obj < 0 ? "NaN" : to_string(obj)) << ","
                << (mse < 0 ? "NaN" : to_string(mse)) << ","
                << elapsed() << "\n";
            log.flush();
        } // end iteration loop

        // Final decrypt for this experiment
        bcastU32(fds, MAGIC_END);
        bcast(fds, encZ, cc);
        vector<Ciphertext<DCRTPoly>> fsh(NUM_PARTIES);
        for (int i = 0; i < NUM_PARTIES; i++)
            fsh[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
        try {
            Plaintext ptF; cc->MultipartyDecryptFusion(fsh, &ptF);
            ptF->SetLength(N_FEAT);
            auto zF = ptF->GetRealPackedValue(); zF.resize(N_FEAT);
            double finalObj = computeObjective(zF, A_full, b_full);
            double finalMSE = computeMSE(zF, x_true);
            cout << "  Sparsity " << pct << "% final obj=" << finalObj
                 << "  mse=" << finalMSE
                 << (exploded ? " [EXPLODED]" : "") << endl;
        } catch (...) {
            cout << "  [FINAL_DECRYPT_FAIL] sparsity " << pct << "%" << endl;
        }
        log.close();
        cout << "  Log saved: " << log_fname << endl;

    } // end sparsity loop

    // Signal all done
    bcastU32(fds, MAGIC_ALL_DONE);
    cout << "\n=== All sparsity experiments complete. Total time: "
         << fixed << setprecision(1) << elapsed() << "s ===" << endl;

    for (int i = 0; i < NUM_PARTIES; i++) netClose(fds[i]);
    return 0;
}
