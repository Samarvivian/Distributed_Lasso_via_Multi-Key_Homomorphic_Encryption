/**
 * @file server.cpp
 * @brief TRIAD Protocol - Server (runs on the laptop/workstation)
 *
 * Deployment - Distributed mode (Raspberry Pi):
 *   1. Run keygen on this machine first (once):  ./keygen
 *   2. Distribute keys to Pi clients:
 *        for i in 0 1 2; do
 *          scp keys/cryptocontext.bin keys/joint_pk.bin keys/sk_${i}.bin \
 *              pi${i}:~/triad/keys/
 *        done
 *   3. Start clients on each Pi:
 *        Pi0: ./client 0
 *        Pi1: ./client 1
 *        Pi2: ./client 2
 *   4. Start server: ./server
 *
 * Deployment - Local multi-process mode (single machine):
 *   Set CLIENT_IPS all to "127.0.0.1", then: bash run_local.sh
 *
 * NUM_PARTIES = 3
 * Server does NOT hold any secret key.
 *
 * OPTIMIZATIONS vs previous version:
 *   - Lazy u-decrypt: step (d) maskedDecryptSend for u_i is now skipped in
 *     most iterations. It only fires in the iteration immediately before a
 *     Refresh, so clients can rebuild their plaintext ui for re-encryption.
 *     Saves ~1 full masked-decrypt round-trip (~K×RTT) per normal iteration.
 *   - side-decrypt interval: objective monitoring every SIDE_DECRYPT_INTERVAL
 *     iters instead of every iter (was already done in this version).
 *
 * Protocol signals added:
 *   MAGIC_DO_U_DECRYPT   (0x00000006) — server tells clients: decrypt u this iter
 *   MAGIC_SKIP_U_DECRYPT (0x00000007) — server tells clients: skip u decrypt
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

// ============================================================================
// Cross-platform socket
// ============================================================================
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
// Config — must match keygen.cpp and client.cpp EXACTLY
// ============================================================================
static const int    NUM_PARTIES    = 3;
static const int    BASE_PORT      = 10000;
static const size_t N_FEAT         = 200;
static const size_t M_ROWS         = 50;
static const double rho            = 1.0;
static const double lambda_lasso   = 0.1;
static const int    maxIter        = 50;
static const int    updateInterval = 5;
static const int    shrinkWarmup   = 5;
static const int    chebyDegree    = 13;
static const double alpha_crc      = 1.2;
static const double gamma_smooth   = 0.8;
static const double delta_safe     = 0.95;
static const int    Lmin_levels    = 6;
static const double kappa          = lambda_lasso / (rho * NUM_PARTIES);

// How many levels does one Chebyshev call consume?
// deg=13 → ceil(log2(13+1))=4 squarings + a few mults ≈ 8 levels consumed.
// We pre-trigger u-decrypt one iteration before remLev would drop below Lmin.
static const int    REFRESH_LOOKAHEAD = 1;   // trigger u-decrypt this many iters before refresh

// Pi deployment IPs — set all to "127.0.0.1" for local mode
static const char* CLIENT_IPS[3] = {
    "192.168.186.223",
    "192.168.186.33",
    "192.168.186.214"
};

static const uint32_t MAGIC_READY         = 0xCAFEBABE;
static const uint32_t MAGIC_REFRESH       = 0xABCD1234;
static const uint32_t MAGIC_ITERDONE      = 0x00000001;
static const uint32_t MAGIC_END           = 0xFFFFFFFF;
static const uint32_t MAGIC_PLAIN_ADMM    = 0xAAAAAAAA;
static const uint32_t MAGIC_TRIAD_START   = 0xBBBBBBBB;
static const uint32_t MAGIC_STATIC_R      = 0xCCCCCCCC;
static const uint32_t MAGIC_ADAPTIVE      = 0xDDDDDDDD;
static const uint32_t MAGIC_ALL_DONE      = 0xEEEEEEEE;
static const uint32_t MAGIC_ABORT_SR      = 0xCC1EAF01;
static const uint32_t MAGIC_CONTINUE_ITER = 0x00000003;
static const uint32_t MAGIC_DO_MONITOR    = 0x00000004;
static const uint32_t MAGIC_SKIP_MONITOR  = 0x00000005;
static const uint32_t MAGIC_DO_U_DECRYPT  = 0x00000006;  // NEW: lazy u-decrypt signal
static const uint32_t MAGIC_SKIP_U_DECRYPT= 0x00000007;  // NEW: skip u-decrypt signal

static const int SIDE_DECRYPT_INTERVAL    = 5;

// ============================================================================
// Network helpers
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
static double   recvD  (int fd)             { double v; recvAll(fd, (char*)&v, 8); return v; }
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
// Chebyshev coefficients for soft-threshold on [-R,R], input normalized by R
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
// maskedDecryptSend: server masks enc_val with enc_r, broadcasts to all,
// collects partial decrypt shares, fuses, sends plaintext only to targetId
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
// Data helpers (same seed formula as client.cpp)
// ============================================================================
using Mat = vector<vector<double>>;

static void genData(int pid, Mat& Ai, vector<double>& bi) {
    mt19937 rng(42 + pid * 1000);
    normal_distribution<double> nd(0, 1), noise(0, 0.01);
    uniform_real_distribution<double> ud(1, 2);
    Ai.assign(M_ROWS, vector<double>(N_FEAT));
    for (size_t j = 0; j < N_FEAT; j++) {
        double nm = 0;
        for (size_t i = 0; i < M_ROWS; i++) { Ai[i][j] = nd(rng); nm += Ai[i][j]*Ai[i][j]; }
        nm = sqrt(nm);
        for (size_t i = 0; i < M_ROWS; i++) Ai[i][j] /= nm;
    }
    vector<double> xt(N_FEAT, 0);
    for (int k = 0; k < 10; k++) xt[k*(N_FEAT/10)] = ud(rng);
    bi.resize(M_ROWS);
    for (size_t i = 0; i < M_ROWS; i++) {
        bi[i] = 0;
        for (size_t j = 0; j < N_FEAT; j++) bi[i] += Ai[i][j]*xt[j];
        bi[i] += noise(rng);
    }
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

// ============================================================================
// Matrix algebra helpers
// ============================================================================
static Mat transpose(const Mat& A) {
    size_t m = A.size(), n = A[0].size();
    Mat T(n, vector<double>(m));
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++) T[j][i] = A[i][j];
    return T;
}
static Mat matmul(const Mat& A, const Mat& B) {
    size_t m = A.size(), k = B.size(), n = B[0].size();
    Mat C(m, vector<double>(n, 0));
    for (size_t i = 0; i < m; i++)
        for (size_t l = 0; l < k; l++)
            for (size_t j = 0; j < n; j++) C[i][j] += A[i][l] * B[l][j];
    return C;
}
static Mat matadd(const Mat& A, const Mat& B) {
    size_t m = A.size(), n = A[0].size();
    Mat C(m, vector<double>(n));
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++) C[i][j] = A[i][j] + B[i][j];
    return C;
}
static Mat eye(size_t n, double s = 1.0) {
    Mat I(n, vector<double>(n, 0));
    for (size_t i = 0; i < n; i++) I[i][i] = s;
    return I;
}
static Mat invertSPD(Mat A) {
    size_t n = A.size();
    for (size_t i = 0; i < n; i++) A[i][i] += 1e-10;
    Mat L(n, vector<double>(n, 0));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            double s = A[i][j];
            for (size_t k = 0; k < j; k++) s -= L[i][k] * L[j][k];
            L[i][j] = (i == j) ? sqrt(max(s, 1e-12)) : s / L[j][j];
        }
    }
    Mat Li(n, vector<double>(n, 0));
    for (size_t i = 0; i < n; i++) {
        Li[i][i] = 1.0 / L[i][i];
        for (size_t j = 0; j < i; j++) {
            double s = 0;
            for (size_t k = j; k < i; k++) s -= L[i][k] * Li[k][j];
            Li[i][j] = s / L[i][i];
        }
    }
    Mat Inv(n, vector<double>(n, 0));
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            for (size_t k = 0; k < n; k++) Inv[i][j] += Li[k][i] * Li[k][j];
    return Inv;
}
static vector<double> matvec(const Mat& A, const vector<double>& x) {
    vector<double> y(A.size(), 0);
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = 0; j < x.size(); j++) y[i] += A[i][j] * x[j];
    return y;
}
static vector<double> vecadd(const vector<double>& a, const vector<double>& b) {
    vector<double> c(a.size()); for (size_t i = 0; i < a.size(); i++) c[i] = a[i]+b[i]; return c;
}
static vector<double> vecsub(const vector<double>& a, const vector<double>& b) {
    vector<double> c(a.size()); for (size_t i = 0; i < a.size(); i++) c[i] = a[i]-b[i]; return c;
}
static vector<double> vecscale(const vector<double>& a, double s) {
    vector<double> c(a.size()); for (size_t i = 0; i < a.size(); i++) c[i] = a[i]*s; return c;
}
static vector<double> softThreshVec(const vector<double>& w, double th) {
    vector<double> z(w.size());
    for (size_t i = 0; i < w.size(); i++)
        z[i] = (w[i] > th) ? w[i]-th : (w[i] < -th) ? w[i]+th : 0.0;
    return z;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    bool triadOnly = false;
    for (int i = 1; i < argc; i++)
        if (string(argv[i]) == "--triad-only") triadOnly = true;

    auto t0 = chrono::high_resolution_clock::now();
    auto elapsed = [&]() {
        return chrono::duration<double>(chrono::high_resolution_clock::now() - t0).count();
    };

    netInit();
    cout << "=== TRIAD Server (K=" << NUM_PARTIES << ")"
         << (triadOnly ? " [TRIAD-ONLY]" : "") << " ===" << endl;
    cout << "  lambda=" << lambda_lasso << " kappa=" << kappa
         << " Lmin=" << Lmin_levels << endl;

    vector<double> zPlain(N_FEAT, 0.0);

    // -----------------------------------------------------------------------
    // Phase 0: Load keys
    // -----------------------------------------------------------------------
    cout << "\n[Phase 0] Loading keys..." << endl;
    CryptoContext<DCRTPoly> cc;
    Serial::DeserializeFromFile("keys/cryptocontext.bin", cc, SerType::BINARY);
    cc->Enable(PKE); cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE); cc->Enable(ADVANCEDSHE); cc->Enable(MULTIPARTY);
    cout << "  CryptoContext loaded (t=" << elapsed() << "s)" << endl;

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

    Mat globalA; vector<double> globalB;
    for (int pid = 0; pid < NUM_PARTIES; pid++) {
        Mat Ai; vector<double> bi; genData(pid, Ai, bi);
        for (auto& row : Ai) globalA.push_back(row);
        for (double v : bi) globalB.push_back(v);
    }

    // -----------------------------------------------------------------------
    // Connect
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
    // Distributed Plaintext ADMM baseline
    // -----------------------------------------------------------------------
    if (!triadOnly) {
    cout << "\n[Plaintext ADMM] Starting distributed baseline..." << endl;
    bcastU32(fds, MAGIC_PLAIN_ADMM);

    {
        vector<double> zP(N_FEAT, 0.0);
        vector<vector<double>> uP(NUM_PARTIES, vector<double>(N_FEAT, 0.0));

        ofstream plog("plaintext_admm_log.csv");
        plog << "iter,objective,elapsed_s\n";

        for (int iter = 0; iter < maxIter; iter++) {
            for (int i = 0; i < NUM_PARTIES; i++)
                sendVec(fds[i], vecsub(zP, uP[i]));

            vector<vector<double>> xP(NUM_PARTIES);
            for (int i = 0; i < NUM_PARTIES; i++)
                xP[i] = recvVec(fds[i]);

            vector<double> w(N_FEAT, 0.0);
            for (int i = 0; i < NUM_PARTIES; i++)
                w = vecadd(w, vecadd(xP[i], uP[i]));
            double maxW = 0.0;
            for (double v : w) maxW = max(maxW, fabs(v / NUM_PARTIES));
            zP = softThreshVec(vecscale(w, 1.0 / NUM_PARTIES), kappa);

            for (int i = 0; i < NUM_PARTIES; i++)
                uP[i] = vecadd(uP[i], vecsub(xP[i], zP));

            double obj = computeObjective(zP, globalA, globalB);
            cout << "  [PlainADMM] iter=" << setw(2) << iter
                 << "  max|w|=" << fixed << setprecision(4) << maxW
                 << "  obj=" << fixed << setprecision(6) << obj
                 << "  t=" << fixed << setprecision(2) << elapsed() << "s" << endl;
            plog << iter << "," << obj << "," << elapsed() << "\n";
            plog.flush();
        }
        plog.close();
        zPlain = zP;
        cout << "[Plaintext ADMM] Done. Final obj="
             << computeObjective(zPlain, globalA, globalB) << endl;
    }
    } // end PlainADMM

    // -----------------------------------------------------------------------
    // Static-R experiments (disabled — loop body is skipped via continue)
    // -----------------------------------------------------------------------
    struct SRResult { string label; double finalObj; bool exploded; };
    vector<SRResult> srResults;
    if (!triadOnly) {
    srResults.push_back({"Plaintext_ADMM",
                         computeObjective(zPlain, globalA, globalB), false});

    const vector<double> staticRs = {0.5, 1.5, 2.0, 3.0, 5.0, 10.0};
    for (double sr : staticRs) {
        (void)sr; continue; // Static-R disabled
    }
    } // end Static-R

    // -----------------------------------------------------------------------
    // Signal TRIAD start
    // -----------------------------------------------------------------------
    bcastU32(fds, MAGIC_TRIAD_START);
    cout << "  MAGIC_TRIAD_START sent (t=" << elapsed() << "s)" << endl;

    // -----------------------------------------------------------------------
    // Phase 1: Bootstrap R
    // -----------------------------------------------------------------------
    cout << "\n[Phase 1] Bootstrap R..." << endl;
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
    cout << "  R^(0)=" << currentR << " sumSq=" << sumSq
         << " kappa=" << kappa << " (t=" << elapsed() << "s)" << endl;

    auto ptZero = cc->MakeCKKSPackedPlaintext(vector<double>(N_FEAT, 0.0));
    auto encZ   = cc->Encrypt(jointPK, ptZero);
    vector<Ciphertext<DCRTPoly>> encU(NUM_PARTIES);
    for (int i = 0; i < NUM_PARTIES; i++)
        encU[i] = cc->Encrypt(jointPK, ptZero);
    auto coeffs = chebyCoeffs(currentR, chebyDegree);

    // -----------------------------------------------------------------------
    // Phase 2: ADMM iterations
    // -----------------------------------------------------------------------
    cout << "\n[Phase 2] Running " << maxIter << " iterations "
         << "(Lmin=" << Lmin_levels << " totalLev=" << totalLevels << ")..." << endl;

    ofstream log("Adaptive_TRIAD_log.csv");
    log << "iter,R,remLev,crc,refresh,u_decrypt,objective,mse,"
           "cpu_temp_C,mem_used_MB,net_MBps_client,cpu_freq_MHz,elapsed_s\n";
    ofstream tlog("timing_log.csv");
    tlog << "iter,t_mask_ms,t_vdecrypt_ms,t_xcollect_ms,t_udecrypt_ms,t_wupdate_ms"
            ",t_crc_ms,t_cheby_ms,t_broadcast_ms,t_refresh_ms,t_sidedecrypt_ms,t_total_ms"
            ",t_compute_ms,t_comm_normal_ms,t_comm_crc_ms,t_wait_ms,crc_overhead_pct\n";

    struct REvent { int iter; double oldR; double R_raw; double newR; double Psi; bool safeAll; };
    vector<REvent> rHistory;
    rHistory.push_back({-1, 0.0, 0.0, currentR, 0.0, true});

    using hrc = chrono::high_resolution_clock;
    auto hms = [](hrc::time_point a, hrc::time_point b) {
        return chrono::duration<double, milli>(b - a).count();
    };
    vector<double> v_t_mask, v_t_vdecrypt, v_t_xcollect, v_t_udecrypt,
                   v_t_wupdate, v_t_crc, v_t_cheby, v_t_broadcast,
                   v_t_refresh, v_t_sidedecrypt, v_t_total;

    // Client needs ui plaintext only when a Refresh is about to happen.
    // We predict refresh by checking if remLev after this iter's Chebyshev
    // will fall below Lmin_levels. Since we know chebyDegree, we can estimate
    // how many levels the next EvalChebyshevSeries will consume and pre-send
    // u REFRESH_LOOKAHEAD iterations ahead.
    //
    // Simpler and robust approach: send u decrypt whenever
    //   remLev - cheby_cost <= Lmin_levels + REFRESH_LOOKAHEAD
    // where cheby_cost is estimated conservatively as ceil(log2(chebyDegree+1))*2.
    const int cheby_level_cost = 2 * (int)ceil(log2((double)(chebyDegree + 1)));
    cout << "  Estimated Chebyshev level cost per iter: " << cheby_level_cost << endl;

    for (int iter = 0; iter < maxIter; iter++) {
        auto tp_start = hrc::now();
        bool did_crc = false, did_ref = false, did_u_decrypt = false;
        size_t remLevels = totalLevels - encZ->GetLevel() - 1;
        cout << "  iter=" << setw(2) << iter
             << "  R=" << fixed << setprecision(4) << currentR
             << "  remLev=" << remLevels
             << "  t=" << fixed << setprecision(1) << elapsed() << "s" << endl;
        cout.flush();

        // ── Decide whether to do u-decrypt this iteration ──────────────────
        // We need ui plaintext BEFORE the Refresh call in step (i).
        // A Refresh fires when remLev (after Chebyshev) < Lmin_levels.
        // remLev after Chebyshev ≈ remLevels - cheby_level_cost.
        // So fire u-decrypt whenever that predicted value is within
        // REFRESH_LOOKAHEAD of Lmin_levels, i.e.:
        //   remLevels - cheby_level_cost <= Lmin_levels + REFRESH_LOOKAHEAD
        bool need_u_decrypt = ((int)remLevels - cheby_level_cost
                                <= Lmin_levels + REFRESH_LOOKAHEAD);

        // Also always do u-decrypt on iter 0 so clients have initial ui=0
        // confirmed (not strictly necessary since ui starts at 0, but safe).
        // Uncomment if you want extra safety:
        // need_u_decrypt |= (iter == 0);

        // a) recv masks from all clients
        // Clients always send both enc(r_v) and enc(r_u).
        // Server only uses enc(r_u) when need_u_decrypt is true.
        vector<Ciphertext<DCRTPoly>> encRv(NUM_PARTIES), encRu(NUM_PARTIES);
        for (int i = 0; i < NUM_PARTIES; i++) {
            encRv[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
            encRu[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
        }
        auto tp_a = hrc::now();

        // b) masked decrypt ALL v_i  (always required for x-update)
        for (int i = 0; i < NUM_PARTIES; i++)
            maskedDecryptSend(i, fds, cc->EvalSub(encZ, encU[i]), encRv[i], cc, N_FEAT);
        auto tp_b = hrc::now();

        // c) collect enc(x_i)
        vector<Ciphertext<DCRTPoly>> encX(NUM_PARTIES);
        for (int i = 0; i < NUM_PARTIES; i++)
            encX[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
        auto tp_c = hrc::now();

        // d) LAZY u-decrypt: only when a Refresh is imminent
        //    Signal clients first so they know whether to expect the decrypt.
        if (need_u_decrypt) {
            did_u_decrypt = true;
            bcastU32(fds, MAGIC_DO_U_DECRYPT);
            for (int i = 0; i < NUM_PARTIES; i++)
                maskedDecryptSend(i, fds, encU[i], encRu[i], cc, N_FEAT);
            cout << "    [d] u masked-decrypts done (pre-refresh)" << endl;
        } else {
            bcastU32(fds, MAGIC_SKIP_U_DECRYPT);
            // enc(r_u) that clients sent are discarded — no RTT spent.
        }
        auto tp_d = hrc::now();

        // e) enc(w) = (1/K)*sum(enc(x_i)+enc(u_i))
        auto encW = cc->EvalAdd(encX[0], encU[0]);
        for (int i = 1; i < NUM_PARTIES; i++)
            encW = cc->EvalAdd(encW, cc->EvalAdd(encX[i], encU[i]));
        encW = cc->EvalMult(encW, 1.0 / NUM_PARTIES);
        auto tp_e = hrc::now();

        // f) CRC
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
            } else if (!safeAll) {
                currentR = currentR / delta_safe;
            } else if (iter > shrinkWarmup && safeAll) {
                currentR = max(R_raw, gamma_smooth * currentR);
            }
            coeffs = chebyCoeffs(currentR, chebyDegree);
            rHistory.push_back({iter, oldR, R_raw, currentR, Psi, safeAll});
            cout << "    CRC Psi=" << Psi << " R_raw=" << R_raw
                 << " R=" << currentR << " safe=" << safeAll
                 << " dR=" << (currentR-oldR) << endl;
        }
        auto tp_f = hrc::now();

        // g) Chebyshev z-update
        auto encWn   = cc->EvalMult(encW, 1.0 / currentR);
        auto encZnew = cc->EvalChebyshevSeries(encWn, coeffs, -1.0, 1.0);
        auto tp_g = hrc::now();

        // h) broadcast enc(z), collect enc(u_i)
        bcast(fds, encZnew, cc);
        for (int i = 0; i < NUM_PARTIES; i++)
            encU[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
        encZ = encZnew;
        auto tp_h = hrc::now();

        // i) Refresh if needed
        remLevels = totalLevels - encZ->GetLevel() - 1;
        if ((int)remLevels < Lmin_levels) {
            did_ref = true;
            cout << "    [Refresh] remLev=" << remLevels << endl;
            bcastU32(fds, MAGIC_REFRESH);
            bcast(fds, encZ, cc);
            vector<Ciphertext<DCRTPoly>> zsh(NUM_PARTIES);
            for (int i = 0; i < NUM_PARTIES; i++)
                zsh[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
            Plaintext ptZp; cc->MultipartyDecryptFusion(zsh, &ptZp); ptZp->SetLength(N_FEAT);
            encZ = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(ptZp->GetRealPackedValue()));
            bcast(fds, encZ, cc);
            for (int i = 0; i < NUM_PARTIES; i++)
                encU[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
            cout << "    [Refresh] done, remLev=" << (totalLevels-encZ->GetLevel()-1) << endl;
        } else {
            bcastU32(fds, MAGIC_ITERDONE);
        }
        auto tp_i = hrc::now();

        // j) side-decrypt for objective monitoring (sparse)
        double obj = -1.0, mse = -1.0;
        bool do_monitor = (iter % SIDE_DECRYPT_INTERVAL == 0) || (iter == maxIter - 1);
        if (do_monitor) {
            bcastU32(fds, MAGIC_DO_MONITOR);
            bcast(fds, encZ, cc);
            vector<Ciphertext<DCRTPoly>> objSh(NUM_PARTIES);
            for (int i = 0; i < NUM_PARTIES; i++)
                objSh[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
            try {
                Plaintext ptObj; cc->MultipartyDecryptFusion(objSh, &ptObj);
                ptObj->SetLength(N_FEAT);
                auto zvec = ptObj->GetRealPackedValue(); zvec.resize(N_FEAT);
                obj = computeObjective(zvec, globalA, globalB);
                mse = 0.0;
                for (size_t i = 0; i < globalA.size(); i++) {
                    double Az = 0.0;
                    for (size_t j = 0; j < N_FEAT; j++) Az += globalA[i][j]*zvec[j];
                    double di = Az - globalB[i]; mse += di*di;
                }
                mse = sqrt(mse / globalA.size());
                cout << "    obj=" << fixed << setprecision(6) << obj
                     << "  mse=" << mse << endl;
            } catch (...) { cout << "    obj=N/A" << endl; }
        } else {
            bcastU32(fds, MAGIC_SKIP_MONITOR);
        }

        // j2) Collect per-iteration system metrics from Pi clients
        double avg_temp = 0.0, avg_mem = 0.0, avg_net_MBps = 0.0, avg_freq = 0.0;
        for (int i = 0; i < NUM_PARTIES; i++) {
            avg_temp += recvD(fds[i]);
            avg_mem  += recvD(fds[i]);
            avg_net_MBps += recvD(fds[i]);
            avg_freq += recvD(fds[i]);
        }
        avg_temp /= NUM_PARTIES; avg_mem /= NUM_PARTIES;
        avg_net_MBps /= NUM_PARTIES; avg_freq /= NUM_PARTIES;
        auto tp_j = hrc::now();

        double dt_mask        = hms(tp_start, tp_a);
        double dt_vdecrypt    = hms(tp_a,     tp_b);
        double dt_xcollect    = hms(tp_b,     tp_c);
        double dt_udecrypt    = hms(tp_c,     tp_d);
        double dt_wupdate     = hms(tp_d,     tp_e);
        double dt_crc         = hms(tp_e,     tp_f);
        double dt_cheby       = hms(tp_f,     tp_g);
        double dt_broadcast   = hms(tp_g,     tp_h);
        double dt_refresh     = hms(tp_h,     tp_i);
        double dt_sidedecrypt = hms(tp_i,     tp_j);
        double dt_total       = hms(tp_start, tp_j);

        v_t_mask.push_back(dt_mask);
        v_t_vdecrypt.push_back(dt_vdecrypt);
        v_t_xcollect.push_back(dt_xcollect);
        v_t_udecrypt.push_back(dt_udecrypt);
        v_t_wupdate.push_back(dt_wupdate);
        v_t_crc.push_back(dt_crc);
        v_t_cheby.push_back(dt_cheby);
        v_t_broadcast.push_back(dt_broadcast);
        v_t_refresh.push_back(dt_refresh);
        v_t_sidedecrypt.push_back(dt_sidedecrypt);
        v_t_total.push_back(dt_total);

        log << iter << "," << currentR << ","
            << (totalLevels-encZ->GetLevel()-1) << ","
            << did_crc << "," << did_ref << "," << did_u_decrypt << ","
            << (obj<0?"NaN":to_string(obj)) << ","
            << (mse<0?"NaN":to_string(mse)) << ","
            << fixed << setprecision(2)
            << avg_temp << "," << avg_mem << ","
            << avg_net_MBps << "," << avg_freq << ","
            << elapsed() << "\n";
        log.flush();
        double comm_normal_ms = dt_mask + dt_xcollect + dt_broadcast;
        double comm_crc_ms    = did_crc ? dt_crc : 0.0;
        double comp_ms        = dt_wupdate + dt_cheby;
        double wait_ms        = dt_vdecrypt + dt_udecrypt + dt_refresh + dt_sidedecrypt;
        double crc_pct        = dt_total > 0.0 ? 100.0 * comm_crc_ms / dt_total : 0.0;

        tlog << iter << fixed << setprecision(3)
             << "," << dt_mask << "," << dt_vdecrypt << "," << dt_xcollect
             << "," << dt_udecrypt << "," << dt_wupdate << "," << dt_crc
             << "," << dt_cheby << "," << dt_broadcast << "," << dt_refresh
             << "," << dt_sidedecrypt << "," << dt_total
             << "," << comp_ms << "," << comm_normal_ms << "," << comm_crc_ms
             << "," << wait_ms << "," << setprecision(2) << crc_pct << "\n";
        tlog.flush();

        // ── Latency breakdown snapshot at iter 20, 30, 40 ──────────────────
        if (iter == 20 || iter == 30 || iter == 40) {
            cout << "\n  ┌─── Latency Snapshot @ iter=" << iter
                 << " (total=" << fixed << setprecision(1) << dt_total << " ms) ───┐" << endl;
            cout << "  │  Compute   (encW+Cheby)         : " << setw(8) << fixed << setprecision(1)
                 << comp_ms << " ms  (" << setw(5) << fixed << setprecision(1)
                 << 100.0*comp_ms/dt_total << "%)" << endl;
            cout << "  │  Comm-norm (masks+xcollect+bcast): " << setw(8) << fixed << setprecision(1)
                 << comm_normal_ms << " ms  (" << setw(5) << fixed << setprecision(1)
                 << 100.0*comm_normal_ms/dt_total << "%)" << endl;
            cout << "  │  Comm-CRC  (extra CRC round-trip): " << setw(8) << fixed << setprecision(1)
                 << comm_crc_ms << " ms  (" << setw(5) << fixed << setprecision(2)
                 << crc_pct << "%)  <-- CRC overhead" << endl;
            cout << "  │  Wait      (decrypt+refresh+side): " << setw(8) << fixed << setprecision(1)
                 << wait_ms << " ms  (" << setw(5) << fixed << setprecision(1)
                 << 100.0*wait_ms/dt_total << "%)" << endl;
            cout << "  │    ├ v-masked-decrypt            : " << setw(8) << fixed << setprecision(1)
                 << dt_vdecrypt << " ms" << endl;
            cout << "  │    ├ u-masked-decrypt            : " << setw(8) << fixed << setprecision(1)
                 << dt_udecrypt << " ms" << endl;
            cout << "  │    ├ Refresh                     : " << setw(8) << fixed << setprecision(1)
                 << dt_refresh << " ms" << endl;
            cout << "  │    └ Side-decrypt                : " << setw(8) << fixed << setprecision(1)
                 << dt_sidedecrypt << " ms" << endl;
            cout << "  └─────────────────────────────────────────────────────────────┘" << endl;
        }
    }

    // -----------------------------------------------------------------------
    // Phase 3: Final result
    // -----------------------------------------------------------------------
    cout << "\n[Phase 3] Final decrypt..." << endl;
    bcastU32(fds, MAGIC_END);
    bcast(fds, encZ, cc);
    vector<Ciphertext<DCRTPoly>> fsh(NUM_PARTIES);
    for (int i = 0; i < NUM_PARTIES; i++)
        fsh[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
    Plaintext ptF; cc->MultipartyDecryptFusion(fsh, &ptF);
    ptF->SetLength(N_FEAT);
    auto zF = ptF->GetRealPackedValue();
    for (int i = 0; i < NUM_PARTIES; i++) sendVec(fds[i], zF);

    cout << "Final z[0..9]:";
    for (int i = 0; i < 10; i++) cout << " " << fixed << setprecision(4) << zF[i];
    cout << "\nFinal objective: " << computeObjective(zF, globalA, globalB) << endl;
    cout << "Total time: " << elapsed() << "s" << endl;

    srResults.push_back({"Adaptive_TRIAD", computeObjective(zF, globalA, globalB), false});

    // ── Table 1: Objective function ──────────────────────────────────────────
    cout << "\n============================================================" << endl;
    cout << " Table 1: Objective Function Comparison" << endl;
    cout << "============================================================" << endl;
    cout << left << setw(26) << "Algorithm"
         << right << setw(16) << "Final Objective"
         << "  Status" << endl;
    cout << string(55, '-') << endl;
    for (auto& r : srResults) {
        cout << left << setw(26) << r.label
             << right << setw(16) << fixed << setprecision(6) << r.finalObj;
        if (r.exploded) cout << "  [EXPLODED]";
        cout << "\n";
    }
    cout << "============================================================" << endl;

    // ── Table 2: Latency breakdown ───────────────────────────────────────────
    {
        int N = (int)v_t_total.size();
        if (N > 0) {
            auto mean_v = [&](const vector<double>& v) {
                double s = 0; for (double x : v) s += x; return s / v.size();
            };
            double mu_mask  = mean_v(v_t_mask),    mu_vdec = mean_v(v_t_vdecrypt),
                   mu_xcol  = mean_v(v_t_xcollect), mu_udec = mean_v(v_t_udecrypt),
                   mu_wup   = mean_v(v_t_wupdate),  mu_crc  = mean_v(v_t_crc),
                   mu_cheby = mean_v(v_t_cheby),    mu_bcast= mean_v(v_t_broadcast),
                   mu_ref   = mean_v(v_t_refresh),  mu_side = mean_v(v_t_sidedecrypt),
                   mu_total = mean_v(v_t_total);

            cout << "\n============================================================" << endl;
            cout << " Table 2: Per-Iteration Latency Breakdown (Adaptive TRIAD)" << endl;
            cout << "  (" << N << " iterations, all times in ms)" << endl;
            cout << "============================================================" << endl;
            cout << left << setw(22) << "Step"
                 << right << setw(12) << "Mean (ms)" << endl;
            cout << string(36, '-') << endl;
            auto row = [&](const char* name, double mu) {
                cout << left << setw(22) << name
                     << right << setw(12) << fixed << setprecision(2) << mu << "\n";
            };
            row("(a) recv_masks",       mu_mask);
            row("(b) v_masked_decrypt", mu_vdec);
            row("(c) x_collect",        mu_xcol);
            row("(d) u_decrypt(lazy)",  mu_udec);
            row("(e) w_update",         mu_wup);
            row("(f) CRC",              mu_crc);
            row("(g) Chebyshev",        mu_cheby);
            row("(h) broadcast_z+u",    mu_bcast);
            row("(i) refresh",          mu_ref);
            row("(j) side_decrypt",     mu_side);
            cout << string(36, '-') << endl;
            row("TOTAL",                mu_total);
            cout << "============================================================" << endl;

            double mean_total_s = mu_total / 1000.0;
            double throughput   = (mean_total_s > 0) ? (double)N_FEAT / mean_total_s : 0.0;
            cout << "\n============================================================" << endl;
            cout << " Table 3: Throughput (Adaptive TRIAD)" << endl;
            cout << "============================================================" << endl;
            cout << left << setw(30) << "Mean iteration time"
                 << right << setw(10) << fixed << setprecision(2) << mu_total << " ms\n";
            cout << left << setw(30) << "N_FEAT"
                 << right << setw(10) << N_FEAT << "\n";
            cout << left << setw(30) << "Throughput"
                 << right << setw(10) << fixed << setprecision(2) << throughput << " features/s\n";
            cout << "============================================================" << endl;
        }
    }

    // ── Table 4: R adaptation history ────────────────────────────────────────
    {
        cout << "\n============================================================" << endl;
        cout << " Table 4: R Adaptation History (Adaptive TRIAD)" << endl;
        cout << "============================================================" << endl;
        cout << right << setw(6)  << "iter"
             << setw(10) << "oldR"
             << setw(10) << "R_raw"
             << setw(10) << "newR"
             << setw(12) << "dR"
             << setw(10) << "Psi"
             << setw(7)  << "safe" << "\n";
        cout << string(65, '-') << "\n";
        for (auto& e : rHistory) {
            if (e.iter == -1) {
                cout << right << setw(6)  << "init"
                     << setw(10) << fixed << setprecision(4) << e.oldR
                     << setw(10) << "-"
                     << setw(10) << fixed << setprecision(4) << e.newR
                     << setw(12) << "-"
                     << setw(10) << "-"
                     << setw(7)  << "-" << "\n";
            } else {
                cout << right << setw(6)  << e.iter
                     << setw(10) << fixed << setprecision(4) << e.oldR
                     << setw(10) << fixed << setprecision(4) << e.R_raw
                     << setw(10) << fixed << setprecision(4) << e.newR
                     << setw(12) << fixed << setprecision(4) << (e.newR - e.oldR)
                     << setw(10) << fixed << setprecision(4) << e.Psi
                     << setw(7)  << (e.safeAll ? "yes" : "no") << "\n";
            }
        }
        cout << "============================================================" << endl;
        cout << "  Total CRC updates: " << (rHistory.size() > 0 ? rHistory.size()-1 : 0) << "\n";
    }

    for (int i = 0; i < NUM_PARTIES; i++) netClose(fds[i]);
    log.close();
    tlog.close();
    return 0;
}