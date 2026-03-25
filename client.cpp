/**
 * @file client.cpp
 * @brief TRIAD Protocol - Client (runs on Raspberry Pi or local process)
 *
 * Usage:
 *   ./client 0    (party 0, listens on BASE_PORT+0)
 *   ./client 1    (party 1, listens on BASE_PORT+1)
 *   ./client 2    (party 2, listens on BASE_PORT+2)
 *
 * Prerequisites:
 *   keys/cryptocontext.bin  keys/joint_pk.bin  keys/sk_<id>.bin
 *
 * FIXES vs previous version:
 *   - lambda_lasso = 0.1  (must match server.cpp)
 *   - Protocol order: Phase A (all v) → send x → Phase C (all u)
 *     Client now loops j=0..K-1 for ALL v decrypts before sending x,
 *     then loops j=0..K-1 for ALL u decrypts. Matches server.cpp exactly.
 *   - Objective side-decrypt added at end of each iter (matches server.cpp step j)
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

// ============================================================================
// Cross-platform socket
// ============================================================================
#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
  typedef int ssize_t;
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
// Config — must match keygen.cpp and server.cpp EXACTLY
// ============================================================================
static const int    NUM_PARTIES    = 3;
static const int    BASE_PORT      = 10000;
static const size_t N_FEAT         = 200;
static const size_t M_ROWS         = 50;
static const double rho            = 1.0;
static const double lambda_lasso   = 0.1;   // FIX: must match server.cpp
static const int    maxIter        = 50;
static const int    updateInterval = 5;
static const int    shrinkWarmup   = 5;
static const double delta_safe     = 0.95;
static const double gamma_smooth   = 0.8;
static const int    Lmin_levels    = 6;     // FIX: must match server.cpp
static const double B_mask         = 10.0;

static const uint32_t MAGIC_READY       = 0xCAFEBABE;
static const uint32_t MAGIC_REFRESH     = 0xABCD1234;
static const uint32_t MAGIC_ITERDONE   = 0x00000001;
static const uint32_t MAGIC_END         = 0xFFFFFFFF;
static const uint32_t MAGIC_PLAIN_ADMM  = 0xAAAAAAAA;
static const uint32_t MAGIC_TRIAD_START = 0xBBBBBBBB;
static const uint32_t MAGIC_STATIC_R    = 0xCCCCCCCC;
static const uint32_t MAGIC_ADAPTIVE    = 0xDDDDDDDD;
static const uint32_t MAGIC_ALL_DONE    = 0xEEEEEEEE;
static const uint32_t MAGIC_ABORT_SR    = 0xCC1EAF01;  // early-abort for Static-R experiment
static const uint32_t MAGIC_CONTINUE_ITER = 0x00000003; // proceed with next iteration

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
static void     sendD  (int fd, double   v) { sendAll(fd, (char*)&v, 8); }
static double   recvD  (int fd)             { double v; recvAll(fd, (char*)&v, 8); return v; }
static void     sendU32(int fd, uint32_t v) { uint32_t n = htonl(v); sendAll(fd, (char*)&n, 4); }
static uint32_t recvU32(int fd)             { uint32_t n; recvAll(fd, (char*)&n, 4); return ntohl(n); }
static void     sendBool(int fd, bool v)    { uint8_t b = v; sendAll(fd, (char*)&b, 1); }

// ============================================================================
// Matrix helpers
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
            for (size_t j = 0; j < n; j++) C[i][j] += A[i][l]*B[l][j];
    return C;
}
static Mat matadd(const Mat& A, const Mat& B) {
    size_t m = A.size(), n = A[0].size();
    Mat C(m, vector<double>(n));
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++) C[i][j] = A[i][j]+B[i][j];
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
            for (size_t k = 0; k < j; k++) s -= L[i][k]*L[j][k];
            L[i][j] = (i==j) ? sqrt(max(s,1e-12)) : s/L[j][j];
        }
    }
    Mat Li(n, vector<double>(n, 0));
    for (size_t i = 0; i < n; i++) {
        Li[i][i] = 1.0/L[i][i];
        for (size_t j = 0; j < i; j++) {
            double s = 0;
            for (size_t k = j; k < i; k++) s -= L[i][k]*Li[k][j];
            Li[i][j] = s/L[i][i];
        }
    }
    Mat Inv(n, vector<double>(n, 0));
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            for (size_t k = 0; k < n; k++) Inv[i][j] += Li[k][i]*Li[k][j];
    return Inv;
}
static vector<double> matvec(const Mat& A, const vector<double>& x) {
    size_t m = A.size();
    vector<double> y(m, 0);
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < x.size(); j++) y[i] += A[i][j]*x[j];
    return y;
}
static vector<double> vecadd(const vector<double>& a, const vector<double>& b) {
    vector<double> c(a.size());
    for (size_t i = 0; i < a.size(); i++) c[i] = a[i]+b[i];
    return c;
}
static vector<double> vecscale(const vector<double>& a, double s) {
    vector<double> c(a.size());
    for (size_t i = 0; i < a.size(); i++) c[i] = a[i]*s;
    return c;
}
static vector<double> vecsub(const vector<double>& a, const vector<double>& b) {
    vector<double> c(a.size());
    for (size_t i = 0; i < a.size(); i++) c[i] = a[i]-b[i];
    return c;
}

// ============================================================================
// Deterministic data generation (same seed as server.cpp)
// ============================================================================
static void genData(int pid, Mat& Ai, vector<double>& bi) {
    mt19937 rng(42 + pid * 1000);
    normal_distribution<double> nd(0,1), noise(0,0.01);
    uniform_real_distribution<double> ud(1,2);
    Ai.assign(M_ROWS, vector<double>(N_FEAT));
    for (size_t j = 0; j < N_FEAT; j++) {
        double nm = 0;
        for (size_t i = 0; i < M_ROWS; i++) { Ai[i][j]=nd(rng); nm+=Ai[i][j]*Ai[i][j]; }
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

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) { cerr << "Usage: ./client <party_id 0|1|2> [--triad-only]\n"; return 1; }
    int myId = atoi(argv[1]);
    if (myId < 0 || myId >= NUM_PARTIES) {
        cerr << "party_id must be 0.." << (NUM_PARTIES-1) << "\n"; return 1;
    }
    bool triadOnly = false;
    for (int i = 2; i < argc; i++)
        if (string(argv[i]) == "--triad-only") triadOnly = true;

    auto t0 = chrono::high_resolution_clock::now();
    auto elapsed = [&]() {
        return chrono::duration<double>(chrono::high_resolution_clock::now()-t0).count();
    };

    netInit();
    cout << "=== TRIAD Client " << myId << " ===" << endl;
    cout << "  lambda=" << lambda_lasso << " Lmin=" << Lmin_levels << endl;

    // -----------------------------------------------------------------------
    // Load keys
    // -----------------------------------------------------------------------
    cout << "\n[Init] Loading keys..." << endl;
    CryptoContext<DCRTPoly> cc;
    Serial::DeserializeFromFile("keys/cryptocontext.bin", cc, SerType::BINARY);
    cc->Enable(PKE); cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE); cc->Enable(ADVANCEDSHE); cc->Enable(MULTIPARTY);
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
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(BASE_PORT + myId);
    addr.sin_addr.s_addr = INADDR_ANY;
    bind(lfd, (sockaddr*)&addr, sizeof(addr));
    listen(lfd, 5);
    int fd = accept(lfd, nullptr, nullptr);
    netClose(lfd);
    cout << "  Server connected (t=" << elapsed() << "s)" << endl;
    sendU32(fd, MAGIC_READY);

    // -----------------------------------------------------------------------
    // Precompute M_i, g_i
    // -----------------------------------------------------------------------
    cout << "\n[Precompute] M_i and g_i..." << endl;
    Mat Ai; vector<double> bi;
    genData(myId, Ai, bi);
    auto At  = transpose(Ai);
    auto AtA = matmul(At, Ai);
    auto Mi  = invertSPD(matadd(AtA, eye(N_FEAT, rho)));
    auto gi  = matvec(At, bi);
    cout << "  Done (t=" << elapsed() << "s)" << endl;

    // -----------------------------------------------------------------------
    // Distributed Plaintext ADMM (runs before TRIAD)
    // -----------------------------------------------------------------------
    if (!triadOnly) {
    {
        uint32_t sig = recvU32(fd);
        if (sig != MAGIC_PLAIN_ADMM)
            throw runtime_error("Expected MAGIC_PLAIN_ADMM, got: " + to_string(sig));
        cout << "\n[PlainADMM] Starting plaintext ADMM loop..." << endl;

        for (int iter = 0; iter < maxIter; iter++) {
            cout << "  [PlainADMM] iter=" << setw(2) << iter
                 << "  t=" << fixed << setprecision(1) << elapsed() << "s" << endl;
            cout.flush();
            // Receive v_i = z - u_i from server
            auto vi_plain = recvVec(fd);
            // x-update: x_i = M_i * (g_i + rho * v_i)
            auto xi_plain = matvec(Mi, vecadd(gi, vecscale(vi_plain, rho)));
            // Send x_i back to server
            sendVec(fd, xi_plain);
        }

        // -----------------------------------------------------------------------
        // Static-R experiments: handle any MAGIC_STATIC_R before MAGIC_TRIAD_START
        // -----------------------------------------------------------------------
        while (true) {
            sig = recvU32(fd);
            if (sig == MAGIC_TRIAD_START) {
                cout << "[PlainADMM] Done. Proceeding to TRIAD..." << endl;
                break;
            }
            if (sig != MAGIC_STATIC_R)
                throw runtime_error("Expected MAGIC_STATIC_R or MAGIC_TRIAD_START");

            double fixedR = recvD(fd);
            cout << "\n===== StaticR=" << fixed << setprecision(1) << fixedR
                 << " =====" << endl;

            // Init state — no Bootstrap R, use fixedR directly
            vector<double> uiS(N_FEAT, 0.0);
            auto encUiS = cc->Encrypt(jointPK,
                              cc->MakeCKKSPackedPlaintext(vector<double>(N_FEAT, 0.0)));
            Ciphertext<DCRTPoly> encXiS = cc->Encrypt(jointPK,
                              cc->MakeCKKSPackedPlaintext(vector<double>(N_FEAT, 0.0)));

            // ADMM iterations — same as TRIAD Phase 2 but WITHOUT CRC (step e)
            for (int iter = 0; iter < maxIter; iter++) {
                // Pre-iteration handshake: server signals abort or continue
                uint32_t iterCtrl = recvU32(fd);
                if (iterCtrl == MAGIC_ABORT_SR) {
                    cout << "  [ABORT_SR] server aborted StaticR="
                         << fixed << setprecision(1) << fixedR
                         << " — proceeding to final decrypt" << endl;
                    break;
                }
                // else MAGIC_CONTINUE_ITER — proceed normally

                cout << "  iter=" << setw(2) << iter
                     << "  t=" << fixed << setprecision(1) << elapsed() << "s" << endl;
                cout.flush();

                // a) Send fresh masks
                vector<double> rvS(N_FEAT), ruS(N_FEAT);
                for (auto& v : rvS) v = maskDist(rng);
                for (auto& v : ruS) v = maskDist(rng);
                auto encRvS = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(rvS));
                auto encRuS = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(ruS));
                sendObj(fd, encRvS, cc);
                sendObj(fd, encRuS, cc);

                // b) Phase A: participate in ALL v masked-decrypts
                vector<double> viS(N_FEAT, 0.0);
                for (int j = 0; j < NUM_PARTIES; j++) {
                    auto encVjM = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
                    Ciphertext<DCRTPoly> partV;
                    if (myId == 0) partV = cc->MultipartyDecryptLead({encVjM}, mySK)[0];
                    else           partV = cc->MultipartyDecryptMain({encVjM}, mySK)[0];
                    sendObj(fd, partV, cc);
                    if (j == myId) {
                        auto viM = recvVec(fd);
                        for (size_t k = 0; k < N_FEAT; k++) viS[k] = viM[k] - rvS[k];
                    }
                }

                // c) x-update
                auto xiS = matvec(Mi, vecadd(gi, vecscale(viS, rho)));
                encXiS = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(xiS));
                sendObj(fd, encXiS, cc);

                // d) Phase C: participate in ALL u masked-decrypts
                for (int j = 0; j < NUM_PARTIES; j++) {
                    auto encUjM = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
                    Ciphertext<DCRTPoly> partU;
                    if (myId == 0) partU = cc->MultipartyDecryptLead({encUjM}, mySK)[0];
                    else           partU = cc->MultipartyDecryptMain({encUjM}, mySK)[0];
                    sendObj(fd, partU, cc);
                    if (j == myId) {
                        auto uiM = recvVec(fd);
                        for (size_t k = 0; k < N_FEAT; k++) uiS[k] = uiM[k] - ruS[k];
                    }
                }

                // e) [NO CRC for Static-R]

                // f) Receive enc(z)
                auto encZnewS = recvObj<Ciphertext<DCRTPoly>>(fd, cc);

                // g) u-update
                encUiS = cc->EvalAdd(encUiS, cc->EvalSub(encXiS, encZnewS));
                sendObj(fd, encUiS, cc);

                // h) MAGIC_REFRESH or MAGIC_ITERDONE
                uint32_t flagS = recvU32(fd);
                if (flagS == MAGIC_REFRESH) {
                    auto encZrefS = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
                    Ciphertext<DCRTPoly> partZS;
                    if (myId == 0) partZS = cc->MultipartyDecryptLead({encZrefS}, mySK)[0];
                    else           partZS = cc->MultipartyDecryptMain({encZrefS}, mySK)[0];
                    sendObj(fd, partZS, cc);
                    auto encZfreshS = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
                    (void)encZfreshS;
                    encUiS = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(uiS));
                    sendObj(fd, encUiS, cc);
                }

                // i) Side-decrypt for objective monitoring
                {
                    auto encZobjS = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
                    Ciphertext<DCRTPoly> partObjS;
                    if (myId == 0) partObjS = cc->MultipartyDecryptLead({encZobjS}, mySK)[0];
                    else           partObjS = cc->MultipartyDecryptMain({encZobjS}, mySK)[0];
                    sendObj(fd, partObjS, cc);
                }
                // i2) Send mem_used_MB
                {
                    double mem_used = 0.0;
#if defined(__linux__)
                    {
                        ifstream f("/proc/meminfo"); string line;
                        long total=0, avail=0;
                        while(getline(f,line)) {
                            if(line.rfind("MemTotal:",0)==0)  sscanf(line.c_str(),"MemTotal: %ld",&total);
                            if(line.rfind("MemAvailable:",0)==0) sscanf(line.c_str(),"MemAvailable: %ld",&avail);
                        }
                        mem_used = (total-avail)/1024.0;
                    }
#endif
                    sendD(fd, mem_used);
                }
            }

            // Final decrypt for this static-R run
            if (recvU32(fd) != MAGIC_END)
                throw runtime_error("Expected MAGIC_END after static-R");
            auto encZfinalS = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
            Ciphertext<DCRTPoly> partFinalS;
            if (myId == 0) partFinalS = cc->MultipartyDecryptLead({encZfinalS}, mySK)[0];
            else           partFinalS = cc->MultipartyDecryptMain({encZfinalS}, mySK)[0];
            sendObj(fd, partFinalS, cc);
            auto zFinalS = recvVec(fd); zFinalS.resize(N_FEAT);
            cout << "StaticR=" << fixedR << " done. z[0..4]:";
            for (int i = 0; i < 5; i++)
                cout << " " << fixed << setprecision(4) << zFinalS[i];
            cout << "\nt=" << elapsed() << "s" << endl;
        }  // end static-R loop
    }
    } // end if (!triadOnly) — PlainADMM+StaticR
    else {
        // triad-only mode: receive MAGIC_TRIAD_START directly
        uint32_t sig = recvU32(fd);
        if (sig != MAGIC_TRIAD_START)
            throw runtime_error("Expected MAGIC_TRIAD_START in triad-only mode");
        cout << "[TRIAD-only] Received MAGIC_TRIAD_START, proceeding to Phase 1..." << endl;
    }

    // -----------------------------------------------------------------------
    // Phase 1: Bootstrap R
    // -----------------------------------------------------------------------
    cout << "\n[Phase 1] Bootstrap R..." << endl;
    auto xi    = matvec(Mi, gi);
    auto encXi = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(xi));
    sendObj(fd, encXi, cc);

    auto ct_sumSq = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
    Ciphertext<DCRTPoly> partSq;
    if (myId == 0) partSq = cc->MultipartyDecryptLead({ct_sumSq}, mySK)[0];
    else           partSq = cc->MultipartyDecryptMain({ct_sumSq}, mySK)[0];
    sendObj(fd, partSq, cc);

    double currentR = recvD(fd);
    cout << "  R^(0)=" << currentR << " (t=" << elapsed() << "s)" << endl;

    vector<double> ui(N_FEAT, 0.0);
    auto encUi = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(vector<double>(N_FEAT, 0.0)));
    Ciphertext<DCRTPoly> encXi_curr = encXi;

    // -----------------------------------------------------------------------
    // Phase 2: ADMM iterations
    // -----------------------------------------------------------------------
    cout << "\n[Phase 2] ADMM iterations..." << endl;

    for (int iter = 0; iter < maxIter; iter++) {
        cout << "  iter=" << setw(2) << iter
             << "  t=" << fixed << setprecision(1) << elapsed() << "s" << endl;
        cout.flush();

        // a) Send fresh masks enc(r_v), enc(r_u)
        vector<double> rv(N_FEAT), ru(N_FEAT);
        for (auto& v : rv) v = maskDist(rng);
        for (auto& v : ru) v = maskDist(rng);
        auto encRv = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(rv));
        auto encRu = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(ru));
        sendObj(fd, encRv, cc);
        sendObj(fd, encRu, cc);
        cout << "    [a] masks sent" << endl; cout.flush();

        // b) Phase A: participate in ALL v masked-decrypts (j=0..K-1)
        //    For j==myId: also receive plaintext(v_i + r_v), recover v_i
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
                cout << "    [b] v_i recovered (j=" << j << ")" << endl; cout.flush();
            }
        }

        // c) x-update and send enc(x_i)
        xi = matvec(Mi, vecadd(gi, vecscale(vi, rho)));
        encXi_curr = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(xi));
        sendObj(fd, encXi_curr, cc);
        cout << "    [c] enc(x_i) sent" << endl; cout.flush();

        // d) Phase C: participate in ALL u masked-decrypts (j=0..K-1)
        for (int j = 0; j < NUM_PARTIES; j++) {
            auto encUjMasked = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
            Ciphertext<DCRTPoly> partU;
            if (myId == 0) partU = cc->MultipartyDecryptLead({encUjMasked}, mySK)[0];
            else           partU = cc->MultipartyDecryptMain({encUjMasked}, mySK)[0];
            sendObj(fd, partU, cc);
            if (j == myId) {
                auto uiMasked = recvVec(fd);
                for (size_t k = 0; k < N_FEAT; k++) ui[k] = uiMasked[k] - ru[k];
                cout << "    [d] u_i recovered (j=" << j << ")" << endl; cout.flush();
            }
        }

        // e) CRC safety check (if triggered this iter)
        if (iter >= shrinkWarmup && iter % updateInterval == 0) {
            auto ct_Psi = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
            Ciphertext<DCRTPoly> partPsi;
            if (myId == 0) partPsi = cc->MultipartyDecryptLead({ct_Psi}, mySK)[0];
            else           partPsi = cc->MultipartyDecryptMain({ct_Psi}, mySK)[0];
            sendObj(fd, partPsi, cc);
            double R_raw = recvD(fd);
            auto wi = vecadd(xi, ui);
            double maxW = 0;
            for (auto v : wi) maxW = max(maxW, abs(v));
            bool safe = (maxW <= delta_safe * currentR);  // 用currentR而不是R_raw
            sendBool(fd, safe);
            // 同步更新currentR，与server保持一致
            double oldR = currentR;
            if (R_raw > currentR) {
                currentR = R_raw;
            } else if (!safe) {
                currentR = currentR / delta_safe;
            } else if (iter > shrinkWarmup && safe) {
                currentR = max(R_raw, gamma_smooth * currentR);
            }
            cout << "    [e] CRC R_raw=" << R_raw
                 << " R: " << oldR << " -> " << currentR << endl;
        }

        // f) Receive enc(z^(k+1))
        auto encZnew = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
        cout << "    [f] enc(z) received" << endl; cout.flush();

        // g) Homomorphic u-update: enc(u_i) += enc(x_i) - enc(z)
        encUi = cc->EvalAdd(encUi, cc->EvalSub(encXi_curr, encZnew));
        sendObj(fd, encUi, cc);
        cout << "    [g] enc(u_i) sent" << endl; cout.flush();

        // h) Handle MAGIC_REFRESH or MAGIC_ITERDONE
        uint32_t flag = recvU32(fd);
        if (flag == MAGIC_REFRESH) {
            cout << "    [h] REFRESH" << endl; cout.flush();
            auto encZref = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
            Ciphertext<DCRTPoly> partZ;
            if (myId == 0) partZ = cc->MultipartyDecryptLead({encZref}, mySK)[0];
            else           partZ = cc->MultipartyDecryptMain({encZref}, mySK)[0];
            sendObj(fd, partZ, cc);
            // receive fresh enc(z) — server manages it, we just discard
            auto encZfresh = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
            (void)encZfresh;
            // re-encrypt enc(u_i) from plaintext ui (restores level)
            encUi = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(ui));
            sendObj(fd, encUi, cc);
            cout << "    [h] Refresh done" << endl;
        }
        // MAGIC_ITERDONE: nothing to do

        // i) Side-decrypt for server's objective monitoring (step j in server)
        {
            auto encZobj = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
            Ciphertext<DCRTPoly> partObj;
            if (myId == 0) partObj = cc->MultipartyDecryptLead({encZobj}, mySK)[0];
            else           partObj = cc->MultipartyDecryptMain({encZobj}, mySK)[0];
            sendObj(fd, partObj, cc);
        }
        // i2) Send system metrics: cpu_temp_C, mem_used_MB, net_bytes, cpu_freq_MHz
        {
            // CPU temperature
            double cpu_temp = 0.0;
#if defined(__linux__)
            { ifstream f("/sys/class/thermal/thermal_zone0/temp"); int t=0; if(f>>t) cpu_temp=t/1000.0; }
#endif
            // Memory used (MB)
            double mem_used = 0.0;
#if defined(__linux__)
            {
                ifstream f("/proc/meminfo"); string line;
                long total=0, avail=0;
                while(getline(f,line)) {
                    if(line.rfind("MemTotal:",0)==0)  sscanf(line.c_str(),"MemTotal: %ld",&total);
                    if(line.rfind("MemAvailable:",0)==0) sscanf(line.c_str(),"MemAvailable: %ld",&avail);
                }
                mem_used = (total-avail)/1024.0;
            }
#endif
            // Network bytes (RX+TX on all interfaces except lo)
            double net_bytes = 0.0;
#if defined(__linux__)
            {
                ifstream f("/proc/net/dev"); string line;
                getline(f,line); getline(f,line); // skip 2 header lines
                while(getline(f,line)) {
                    string iface; istringstream ss(line);
                    ss>>iface; if(iface=="lo:") continue;
                    long long rx=0,tx=0,tmp;
                    ss>>rx>>tmp>>tmp>>tmp>>tmp>>tmp>>tmp>>tmp>>tx;
                    net_bytes += rx+tx;
                }
            }
#endif
            // CPU frequency (MHz)
            double cpu_freq = 0.0;
#if defined(__linux__)
            { ifstream f("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"); long hz=0; if(f>>hz) cpu_freq=hz/1000.0; }
#endif
            sendD(fd, cpu_temp);
            sendD(fd, mem_used);
            sendD(fd, net_bytes);
            sendD(fd, cpu_freq);
        }

        cout << "    [iter " << iter << " done] t=" << fixed << setprecision(1)
             << elapsed() << "s" << endl; cout.flush();
    }

    // -----------------------------------------------------------------------
    // Phase 3: Final decrypt
    // -----------------------------------------------------------------------
    cout << "\n[Phase 3] Final decrypt..." << endl;
    if (recvU32(fd) != MAGIC_END) throw runtime_error("Expected MAGIC_END");

    auto encZfinal = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
    Ciphertext<DCRTPoly> partFinal;
    if (myId == 0) partFinal = cc->MultipartyDecryptLead({encZfinal}, mySK)[0];
    else           partFinal = cc->MultipartyDecryptMain({encZfinal}, mySK)[0];
    sendObj(fd, partFinal, cc);

    auto zFinal = recvVec(fd);
    zFinal.resize(N_FEAT);

    cout << "Final z[0..9]:";
    for (int i = 0; i < 10; i++)
        cout << " " << fixed << setprecision(4) << zFinal[i];
    cout << "\nTotal time: " << elapsed() << "s" << endl;

    ofstream out("client" + to_string(myId) + "_result.csv");
    for (size_t i = 0; i < N_FEAT; i++) out << i << "," << zFinal[i] << "\n";
    out.close();

    netClose(fd);
    cout << "Client " << myId << " done." << endl;
    return 0;
}