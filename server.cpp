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
 * FIXES vs previous version:
 *   - lambda_lasso = 0.1  (was 1.0 in server.cpp — caused obj~1e14)
 *   - Lmin_levels  = 6    (was 3  — triggered Refresh too late, noise overflow)
 *   - R^(0) floor  = 5.0  (was missing — near-zero R breaks Cheby approximation)
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
static const double lambda_lasso   = 0.1;   // FIX: was 1.0 → caused obj~1e14
static const int    maxIter        = 50;
static const int    updateInterval = 5;
static const int    shrinkWarmup   = 5;
static const int    chebyDegree    = 15;
static const double alpha_crc      = 1.2;
static const double gamma_smooth   = 0.8;
static const double delta_safe     = 0.95;
static const int    Lmin_levels    = 6;     // FIX: was 3 → refresh earlier
static const double kappa          = lambda_lasso / (rho * NUM_PARTIES);

// Pi deployment IPs — set all to "127.0.0.1" for local mode
static const char* CLIENT_IPS[3] = {
    "192.168.249.223",
    "192.168.249.33",
    "192.168.249.214"
};

static const uint32_t MAGIC_READY    = 0xCAFEBABE;
static const uint32_t MAGIC_REFRESH  = 0xABCD1234;
static const uint32_t MAGIC_ITERDONE = 0x00000001;
static const uint32_t MAGIC_END      = 0xFFFFFFFF;

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
// Main
// ============================================================================
int main() {
    auto t0 = chrono::high_resolution_clock::now();
    auto elapsed = [&]() {
        return chrono::duration<double>(chrono::high_resolution_clock::now() - t0).count();
    };

    netInit();
    cout << "=== TRIAD Server (K=" << NUM_PARTIES << ") ===" << endl;
    cout << "  lambda=" << lambda_lasso << " kappa=" << kappa
         << " Lmin=" << Lmin_levels << endl;

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
    // FIX: floor at 5.0 — near-zero R causes Cheby approx to fail
    double currentR = max({1.5 * sqrt(sumSq / N_FEAT), 3.0 * kappa, 5.0});
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

    ofstream log("server_log.csv");
    log << "iter,R,remLev,crc,refresh,objective,elapsed_s\n";

    for (int iter = 0; iter < maxIter; iter++) {
        bool did_crc = false, did_ref = false;
        size_t remLevels = totalLevels - encZ->GetLevel() - 1;
        cout << "  iter=" << setw(2) << iter
             << "  R=" << fixed << setprecision(4) << currentR
             << "  remLev=" << remLevels
             << "  t=" << fixed << setprecision(1) << elapsed() << "s" << endl;
        cout.flush();

        // a) recv masks
        vector<Ciphertext<DCRTPoly>> encRv(NUM_PARTIES), encRu(NUM_PARTIES);
        for (int i = 0; i < NUM_PARTIES; i++) {
            encRv[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
            encRu[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
        }

        // b) masked decrypt ALL v_i  (clients participate in every round)
        for (int i = 0; i < NUM_PARTIES; i++)
            maskedDecryptSend(i, fds, cc->EvalSub(encZ, encU[i]), encRv[i], cc, N_FEAT);

        // c) collect enc(x_i)
        vector<Ciphertext<DCRTPoly>> encX(NUM_PARTIES);
        for (int i = 0; i < NUM_PARTIES; i++)
            encX[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);

        // d) masked decrypt ALL u_i
        for (int i = 0; i < NUM_PARTIES; i++)
            maskedDecryptSend(i, fds, encU[i], encRu[i], cc, N_FEAT);

        // e) enc(w) = (1/K)*sum(enc(x_i)+enc(u_i))
        auto encW = cc->EvalAdd(encX[0], encU[0]);
        for (int i = 1; i < NUM_PARTIES; i++)
            encW = cc->EvalAdd(encW, cc->EvalAdd(encX[i], encU[i]));
        encW = cc->EvalMult(encW, 1.0 / NUM_PARTIES);

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
            if (R_raw > currentR) currentR = R_raw;
            else if (iter > shrinkWarmup && safeAll)
                currentR = max(R_raw, gamma_smooth * currentR);
            coeffs = chebyCoeffs(currentR, chebyDegree);
            cout << "    CRC Psi=" << Psi << " R_raw=" << R_raw
                 << " R=" << currentR << " safe=" << safeAll
                 << " dR=" << (currentR-oldR) << endl;
        }

        // g) Chebyshev z-update
        auto encWn   = cc->EvalMult(encW, 1.0 / currentR);
        auto encZnew = cc->EvalChebyshevSeries(encWn, coeffs, -1.0, 1.0);

        // h) broadcast enc(z), collect enc(u_i)
        bcast(fds, encZnew, cc);
        for (int i = 0; i < NUM_PARTIES; i++)
            encU[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
        encZ = encZnew;

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

        // j) side-decrypt for objective monitoring
        double obj = -1.0;
        {
            bcast(fds, encZ, cc);
            vector<Ciphertext<DCRTPoly>> objSh(NUM_PARTIES);
            for (int i = 0; i < NUM_PARTIES; i++)
                objSh[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
            try {
                Plaintext ptObj; cc->MultipartyDecryptFusion(objSh, &ptObj);
                ptObj->SetLength(N_FEAT);
                auto zvec = ptObj->GetRealPackedValue(); zvec.resize(N_FEAT);
                obj = computeObjective(zvec, globalA, globalB);
                cout << "    obj=" << fixed << setprecision(6) << obj << endl;
            } catch (...) { cout << "    obj=N/A" << endl; }
        }
        log << iter << "," << currentR << ","
            << (totalLevels-encZ->GetLevel()-1) << ","
            << did_crc << "," << did_ref << ","
            << (obj<0?"NaN":to_string(obj)) << "," << elapsed() << "\n";
        log.flush();
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

    for (int i = 0; i < NUM_PARTIES; i++) netClose(fds[i]);
    log.close();
    return 0;
}