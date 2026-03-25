/**
 * @file client_sparsity.cpp
 * @brief TRIAD Sparsity Sensitivity Experiment — Client (Raspberry Pi)
 *
 * Receives A_i from the server once, then runs 5 sequential encrypted TRIAD
 * experiments (sparsity 10%–50%) using the threshold-CKKS protocol.
 * Each experiment receives b_i, runs Phase 1 (Bootstrap R) and Phase 2
 * (ADMM iterations), and participates in the final multiparty decrypt.
 *
 * Key differences from client.cpp / client_real.cpp:
 *   - No data file loading — A_i is received over the network from the server.
 *   - N_FEAT=200 allows direct 200×200 matrix inversion (no Woodbury needed).
 *   - Outer experiment loop driven by MAGIC_SPARSITY_EXP / MAGIC_ALL_DONE.
 *   - Inner ADMM loop driven by MAGIC_CONTINUE_ITER / MAGIC_ABORT_EXP.
 *   - PORT range: BASE_PORT=10200 (avoids conflict with synthetic=9000, real=10000).
 *
 * Build (on Raspberry Pi, using CMakeLists_sparsity_pi.txt or equivalent):
 *   mkdir build && cd build
 *   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=... ..
 *   make client_sparsity
 *
 * Usage:
 *   ./client_sparsity <party_id 0|1|2>
 *
 * Requires:
 *   keys/  directory shared by all parties (cryptocontext.bin, joint_pk.bin,
 *          eval_mult_key.bin, eval_rot_key.bin, eval_sum_key.bin, sk_<id>.bin)
 */

#include "openfhe/pke/openfhe.h"
#include "openfhe/pke/scheme/ckksrns/ckksrns-ser.h"
#include "openfhe/pke/key/key-ser.h"
#include "openfhe/pke/cryptocontext-ser.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <iomanip>

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
// Config — must match server_sparsity.cpp exactly
// ============================================================================
static const int    NUM_PARTIES    = 3;
static const int    BASE_PORT      = 10200;
static const size_t N_FEAT         = 200;
static const int    M_ROWS         = 50;    // rows per party
static const double rho            = 1.0;
static const double lambda_lasso   = 0.1;
static const int    maxIter        = 50;
static const int    updateInterval = 5;
static const int    shrinkWarmup   = 5;
static const double delta_safe     = 0.95;
static const double gamma_smooth   = 0.8;
static const int    Lmin_levels    = 6;
static const double B_mask         = 10.0;

// Magic constants — must match server_sparsity.cpp
static const uint32_t MAGIC_READY         = 0xCAFEBABE;
static const uint32_t MAGIC_REFRESH       = 0xABCD1234;
static const uint32_t MAGIC_ITERDONE      = 0x00000001;
static const uint32_t MAGIC_END           = 0xFFFFFFFF;
static const uint32_t MAGIC_ALL_DONE      = 0xEEEEEEEE;
static const uint32_t MAGIC_SPARSITY_EXP  = 0xDD000001;
static const uint32_t MAGIC_CONTINUE_ITER = 0x00000003;
static const uint32_t MAGIC_ABORT_EXP     = 0xDD000002;

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
static vector<double> recvVec(int fd) {
    uint32_t n; recvAll(fd, (char*)&n, 4);
    uint32_t bytes = ntohl(n);
    vector<double> v(bytes / 8);
    recvAll(fd, (char*)v.data(), bytes);
    return v;
}
static void sendVec(int fd, const vector<double>& v) {
    uint32_t n = htonl((uint32_t)(v.size() * 8));
    sendAll(fd, (char*)&n, 4);
    sendAll(fd, (char*)v.data(), v.size() * 8);
}
static void     sendU32 (int fd, uint32_t v) { uint32_t n = htonl(v); sendAll(fd, (char*)&n, 4); }
static uint32_t recvU32 (int fd)             { uint32_t n; recvAll(fd, (char*)&n, 4); return ntohl(n); }
static double   recvD   (int fd)             { double v; recvAll(fd, (char*)&v, 8); return v; }
static void     sendBool(int fd, bool v)     { uint8_t b = v ? 1 : 0; sendAll(fd, (char*)&b, 1); }

// ============================================================================
// Matrix helpers (identical to client_real.cpp)
// ============================================================================
using Mat = vector<vector<double>>;

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
    size_t m = A.size();
    vector<double> y(m, 0);
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < x.size(); j++) y[i] += A[i][j] * x[j];
    return y;
}
static vector<double> vecadd(const vector<double>& a, const vector<double>& b) {
    vector<double> c(a.size());
    for (size_t i = 0; i < a.size(); i++) c[i] = a[i] + b[i];
    return c;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: ./client_sparsity <party_id 0|1|2>\n";
        return 1;
    }
    int myId = atoi(argv[1]);
    if (myId < 0 || myId >= NUM_PARTIES) {
        cerr << "party_id must be 0.." << (NUM_PARTIES - 1) << "\n";
        return 1;
    }

    auto t0 = chrono::high_resolution_clock::now();
    auto elapsed = [&]() {
        return chrono::duration<double>(chrono::high_resolution_clock::now() - t0).count();
    };

    netInit();
    cout << "=== TRIAD Sparsity Client " << myId
         << " (port " << (BASE_PORT + myId) << ") ===" << endl;
    cout << "  N_FEAT=" << N_FEAT << "  M_ROWS=" << M_ROWS
         << "  lambda=" << lambda_lasso
         << "  maxIter=" << maxIter << "  Lmin=" << Lmin_levels << endl;

    // -----------------------------------------------------------------------
    // Load crypto context and keys
    // -----------------------------------------------------------------------
    cout << "\n[Init] Loading keys..." << endl;
    CryptoContext<DCRTPoly> cc;
    Serial::DeserializeFromFile("keys/cryptocontext.bin", cc, SerType::BINARY);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(MULTIPARTY);
    cout << "  CryptoContext loaded (t=" << elapsed() << "s)" << endl;

    PublicKey<DCRTPoly> jointPK;
    Serial::DeserializeFromFile("keys/joint_pk.bin", jointPK, SerType::BINARY);

    PrivateKey<DCRTPoly> mySK;
    Serial::DeserializeFromFile("keys/sk_" + to_string(myId) + ".bin", mySK, SerType::BINARY);
    cout << "  sk_" << myId << " loaded (t=" << elapsed() << "s)" << endl;

    mt19937_64 rng(999 + myId * 31337);
    uniform_real_distribution<double> maskDist(-B_mask, B_mask);

    // -----------------------------------------------------------------------
    // Listen for server connection
    // -----------------------------------------------------------------------
    cout << "\n[Net] Listening on port " << (BASE_PORT + myId) << "..." << endl;
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(BASE_PORT + myId);
    addr.sin_addr.s_addr = INADDR_ANY;
    bind(lfd, (sockaddr*)&addr, sizeof(addr));
    listen(lfd, 5);
    int fd = accept(lfd, nullptr, nullptr);
    netClose(lfd);
    cout << "  Server connected (t=" << elapsed() << "s)" << endl;
    sendU32(fd, MAGIC_READY);

    // -----------------------------------------------------------------------
    // Receive A_i from server, precompute M_inv = (A^T A + rho I)^{-1}
    // -----------------------------------------------------------------------
    cout << "\n[Precompute] Receiving A_i from server..." << endl;
    uint32_t nrows = recvU32(fd);
    uint32_t ncols = recvU32(fd);
    if (ncols != N_FEAT)
        throw runtime_error("Column count mismatch: expected " +
                            to_string(N_FEAT) + " got " + to_string(ncols));

    Mat Ai(nrows, vector<double>(ncols));
    for (uint32_t r = 0; r < nrows; r++) Ai[r] = recvVec(fd);
    cout << "  Received A_i: " << nrows << " x " << ncols
         << " (t=" << elapsed() << "s)" << endl;

    // Precompute M_inv = (A_i^T A_i + rho*I)^{-1}  [200×200 direct inversion]
    auto At  = transpose(Ai);               // N_FEAT × nrows
    auto AtA = matmul(At, Ai);              // N_FEAT × N_FEAT
    Mat  M   = AtA;
    for (size_t i = 0; i < N_FEAT; i++) M[i][i] += rho;
    auto M_inv = invertSPD(M);
    cout << "  M_inv precomputed (200x200 direct, t=" << elapsed() << "s)" << endl;

    sendU32(fd, MAGIC_READY);
    cout << "  Ready for experiments." << endl;

    // -----------------------------------------------------------------------
    // Outer experiment loop
    // -----------------------------------------------------------------------
    while (true) {
        uint32_t outer = recvU32(fd);

        if (outer == MAGIC_ALL_DONE) {
            cout << "\n[Done] Received MAGIC_ALL_DONE, exiting." << endl;
            break;
        }
        if (outer != MAGIC_SPARSITY_EXP)
            throw runtime_error("Outer loop: expected MAGIC_SPARSITY_EXP or "
                                "MAGIC_ALL_DONE, got " + to_string(outer));

        // Receive sparsity level and b_i for this experiment
        double sparsity = recvD(fd);
        int    pct      = (int)(sparsity * 100);
        auto   bi       = recvVec(fd);

        cout << "\n===== Experiment: sparsity=" << pct << "% =====" << endl;
        cout << "  Received b_i (" << bi.size() << " rows)  t="
             << elapsed() << "s" << endl;

        // Compute g_i = A_i^T * b_i for this experiment
        auto gi = matvec(At, bi);           // N_FEAT-vector

        // Reset dual variable
        vector<double> ui(N_FEAT, 0.0);
        auto encUi = cc->Encrypt(
            jointPK,
            cc->MakeCKKSPackedPlaintext(vector<double>(N_FEAT, 0.0)));

        sendU32(fd, MAGIC_READY);

        // -------------------------------------------------------------------
        // Phase 1: Bootstrap R
        // -------------------------------------------------------------------
        cout << "\n  [Phase 1] Bootstrap R..." << endl;

        // x_i^(0) = M_inv * g_i  (z=0, u=0 initialisation)
        auto xi       = matvec(M_inv, gi);
        auto encXi    = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(xi));
        sendObj(fd, encXi, cc);

        auto ct_sumSq = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
        Ciphertext<DCRTPoly> partSq;
        if (myId == 0) partSq = cc->MultipartyDecryptLead({ct_sumSq}, mySK)[0];
        else           partSq = cc->MultipartyDecryptMain({ct_sumSq}, mySK)[0];
        sendObj(fd, partSq, cc);

        double currentR = recvD(fd);
        cout << "    R^(0)=" << fixed << setprecision(4) << currentR
             << "  t=" << elapsed() << "s" << endl;

        Ciphertext<DCRTPoly> encXi_curr = encXi;
        int safeStreak = 0;

        // -------------------------------------------------------------------
        // Phase 2: ADMM iterations
        // -------------------------------------------------------------------
        cout << "  [Phase 2] ADMM iterations..." << endl;

        for (int iter = 0; iter < maxIter; iter++) {
            // Pre-iteration handshake: CONTINUE or ABORT
            uint32_t ctrl = recvU32(fd);
            if (ctrl == MAGIC_ABORT_EXP) {
                cout << "    [ABORT_EXP] sparsity=" << pct
                     << "% at iter=" << iter << endl;
                break;
            }
            if (ctrl != MAGIC_CONTINUE_ITER)
                throw runtime_error("Phase 2: expected MAGIC_CONTINUE_ITER, got "
                                    + to_string(ctrl));

            cout << "    iter=" << setw(2) << iter
                 << "  t=" << fixed << setprecision(1) << elapsed() << "s" << endl;
            cout.flush();

            // (a) Generate and send fresh random masks enc(r_v), enc(r_u)
            vector<double> rv(N_FEAT), ru(N_FEAT);
            for (auto& v : rv) v = maskDist(rng);
            for (auto& v : ru) v = maskDist(rng);
            auto encRv = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(rv));
            auto encRu = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(ru));
            sendObj(fd, encRv, cc);
            sendObj(fd, encRu, cc);
            cout << "      [a] masks sent" << endl; cout.flush();

            // (b) Participate in ALL v masked-decrypts; recover own v_i
            vector<double> vi(N_FEAT, 0.0);
            for (int j = 0; j < NUM_PARTIES; j++) {
                auto encVjMasked = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
                Ciphertext<DCRTPoly> partV;
                if (myId == 0) partV = cc->MultipartyDecryptLead({encVjMasked}, mySK)[0];
                else           partV = cc->MultipartyDecryptMain({encVjMasked}, mySK)[0];
                sendObj(fd, partV, cc);
                if (j == myId) {
                    auto viMasked = recvVec(fd);
                    for (size_t k = 0; k < N_FEAT; k++) vi[k] = viMasked[k] - rv[k];
                    cout << "      [b] v_i recovered (j=" << j << ")" << endl;
                    cout.flush();
                }
            }

            // (c) x-update: x_i = M_inv * (g_i + rho * v_i); send enc(x_i)
            vector<double> rhs(N_FEAT);
            for (size_t k = 0; k < N_FEAT; k++) rhs[k] = gi[k] + rho * vi[k];
            xi         = matvec(M_inv, rhs);
            encXi_curr = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(xi));
            sendObj(fd, encXi_curr, cc);
            cout << "      [c] enc(x_i) sent" << endl; cout.flush();

            // (d) Participate in ALL u masked-decrypts; recover own u_i
            for (int j = 0; j < NUM_PARTIES; j++) {
                auto encUjMasked = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
                Ciphertext<DCRTPoly> partU;
                if (myId == 0) partU = cc->MultipartyDecryptLead({encUjMasked}, mySK)[0];
                else           partU = cc->MultipartyDecryptMain({encUjMasked}, mySK)[0];
                sendObj(fd, partU, cc);
                if (j == myId) {
                    auto uiMasked = recvVec(fd);
                    for (size_t k = 0; k < N_FEAT; k++) ui[k] = uiMasked[k] - ru[k];
                    cout << "      [d] u_i recovered (j=" << j << ")" << endl;
                    cout.flush();
                }
            }

            // (e) CRC safety check (only on scheduled iterations)
            if (iter >= shrinkWarmup && iter % updateInterval == 0) {
                auto ct_Psi = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
                Ciphertext<DCRTPoly> partPsi;
                if (myId == 0) partPsi = cc->MultipartyDecryptLead({ct_Psi}, mySK)[0];
                else           partPsi = cc->MultipartyDecryptMain({ct_Psi}, mySK)[0];
                sendObj(fd, partPsi, cc);

                double R_raw = recvD(fd);
                auto   wi    = vecadd(xi, ui);
                double maxW  = 0;
                for (auto v : wi) maxW = max(maxW, fabs(v));
                bool safe = (maxW <= delta_safe * currentR);
                sendBool(fd, safe);

                // Mirror server's R update (must stay identical)
                double oldR = currentR;
                if (R_raw > currentR) {
                    currentR = R_raw;
                    safeStreak = 0;
                } else if (!safe) {
                    currentR = currentR / delta_safe;
                    safeStreak = 0;
                } else if (iter > shrinkWarmup && safe) {
                    safeStreak++;
                    if (safeStreak >= 3)
                        currentR = max(R_raw, gamma_smooth * currentR);
                }

                cout << "      [e] CRC R_raw=" << R_raw
                     << " R: " << oldR << " -> " << currentR
                     << " safe=" << safe
                     << " streak=" << safeStreak << endl;
            } else {
                // Non-CRC: per-iteration safe flag (mirrors server's else branch)
                auto wi   = vecadd(xi, ui);
                double maxW = 0.0;
                for (auto v : wi) maxW = max(maxW, fabs(v));
                bool safe = (maxW <= delta_safe * currentR);
                sendBool(fd, safe);
                if (!safe) {
                    double oldR = currentR;
                    currentR = currentR / delta_safe;
                    safeStreak = 0;
                    cout << "      [safe] unsafe maxW=" << fixed << setprecision(4)
                         << maxW << " R: " << oldR << " -> " << currentR << endl;
                }
            }

            // (f) Receive enc(z^(k+1))
            auto encZnew = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
            cout << "      [f] enc(z) received" << endl; cout.flush();

            // (g) Homomorphic u-update: enc(u_i) += enc(x_i) - enc(z_new)
            encUi = cc->EvalAdd(encUi, cc->EvalSub(encXi_curr, encZnew));
            sendObj(fd, encUi, cc);
            cout << "      [g] enc(u_i) sent" << endl; cout.flush();

            // (h) Handle MAGIC_REFRESH or MAGIC_ITERDONE
            uint32_t flag = recvU32(fd);
            if (flag == MAGIC_REFRESH) {
                cout << "      [h] REFRESH" << endl; cout.flush();
                auto encZref = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
                Ciphertext<DCRTPoly> partZ;
                if (myId == 0) partZ = cc->MultipartyDecryptLead({encZref}, mySK)[0];
                else           partZ = cc->MultipartyDecryptMain({encZref}, mySK)[0];
                sendObj(fd, partZ, cc);
                auto encZfresh = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
                (void)encZfresh;
                // Re-encrypt u_i from plaintext (recovered in step d)
                encUi = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(ui));
                sendObj(fd, encUi, cc);
                cout << "      [h] Refresh done" << endl;
            } else if (flag != MAGIC_ITERDONE) {
                throw runtime_error("Phase 2 step h: unexpected flag " +
                                    to_string(flag));
            }

            // (i) Side-decrypt for server objective monitoring
            {
                auto encZobj = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
                Ciphertext<DCRTPoly> partObj;
                if (myId == 0) partObj = cc->MultipartyDecryptLead({encZobj}, mySK)[0];
                else           partObj = cc->MultipartyDecryptMain({encZobj}, mySK)[0];
                sendObj(fd, partObj, cc);
            }

            cout << "      [iter " << iter << " done] t="
                 << fixed << setprecision(1) << elapsed() << "s" << endl;
            cout.flush();
        } // end inner ADMM loop

        // -------------------------------------------------------------------
        // Final multiparty decrypt for this experiment
        // -------------------------------------------------------------------
        cout << "  [Phase 3] Final decrypt for sparsity=" << pct << "%" << endl;
        uint32_t end_sig = recvU32(fd);
        if (end_sig != MAGIC_END)
            throw runtime_error("Expected MAGIC_END, got " + to_string(end_sig));

        auto encZfinal = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
        Ciphertext<DCRTPoly> partFinal;
        if (myId == 0) partFinal = cc->MultipartyDecryptLead({encZfinal}, mySK)[0];
        else           partFinal = cc->MultipartyDecryptMain({encZfinal}, mySK)[0];
        sendObj(fd, partFinal, cc);

        cout << "  Sparsity=" << pct << "% final decrypt share sent  t="
             << fixed << setprecision(1) << elapsed() << "s" << endl;
    } // end outer experiment loop

    netClose(fd);
    cout << "\nTotal time: " << fixed << setprecision(1)
         << elapsed() << "s" << endl;
    return 0;
}
