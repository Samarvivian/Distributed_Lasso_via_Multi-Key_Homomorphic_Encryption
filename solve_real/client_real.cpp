/**
 * @file client_real.cpp
 * @brief TRIAD Protocol - Client, Riboflavin real-data experiment
 *
 * Data layout:
 *   client 0: rows  0-23  (24 samples)
 *   client 1: rows 24-47  (24 samples)
 *   client 2: rows 48-70  (23 samples)
 *
 * Data files: data/riboflavin_X.csv, data/riboflavin_y.csv
 *
 * Usage:
 *   ./client_real <party_id 0|1|2> [--triad-only]
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
// Config — must match server_real.cpp and keygen_real.cpp EXACTLY
// ============================================================================
static const int    NUM_PARTIES    = 3;
static const int    BASE_PORT      = 10000;
static const size_t N_FEAT         = 4088;
static const int    TOTAL_ROWS     = 71;
static const double rho            = 1.0;
static const double lambda_lasso   = 0.01;
static const int    maxIter        = 100;
static const int    updateInterval = 5;
static const int    shrinkWarmup   = 5;
static const double delta_safe     = 0.95;
static const double gamma_smooth   = 0.8;
static const int    Lmin_levels    = 6;
static const double B_mask         = 10.0;

// rows assigned to each client (must match server_real.cpp)
static const int CLIENT_ROW_START[3] = {0,  24, 48};
static const int CLIENT_ROW_END[3]   = {24, 48, 71};

static const uint32_t MAGIC_READY       = 0xCAFEBABE;
static const uint32_t MAGIC_REFRESH     = 0xABCD1234;
static const uint32_t MAGIC_ITERDONE    = 0x00000001;
static const uint32_t MAGIC_END         = 0xFFFFFFFF;
static const uint32_t MAGIC_PLAIN_ADMM  = 0xAAAAAAAA;
static const uint32_t MAGIC_STATIC_R    = 0xCCCCCCCC;
static const uint32_t MAGIC_ABORT_SR    = 0xCC1EAF01;
static const uint32_t MAGIC_CONTINUE_ITER = 0x00000003;
static const uint32_t MAGIC_TRIAD_START = 0xBBBBBBBB;

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
// Data loading — identical partition logic to server_real.cpp
// ============================================================================
static void loadRiboflavinClient(int pid, Mat& Ai, vector<double>& bi) {
    // Load full X
    ifstream fX("data/riboflavin_X.csv");
    if (!fX.is_open()) throw runtime_error("Cannot open data/riboflavin_X.csv");
    string line;
    getline(fX, line);  // skip header
    Mat fullX;
    while (getline(fX, line)) {
        istringstream ss(line);
        string tok;
        vector<double> row;
        bool firstToken = true;
        while (getline(ss, tok, ',')) {
            if (firstToken) {
                firstToken = false;
                bool isNum = !tok.empty() &&
                             (isdigit((unsigned char)tok[0]) || tok[0]=='-' || tok[0]=='.');
                if (!isNum) continue;
            }
            try { row.push_back(stod(tok)); } catch (...) {}
        }
        if (!row.empty()) fullX.push_back(row);
    }
    fX.close();
    if ((int)fullX.size() != TOTAL_ROWS)
        throw runtime_error("riboflavin_X.csv: expected " + to_string(TOTAL_ROWS) +
                            " rows, got " + to_string(fullX.size()));

    // Load full y
    ifstream fY("data/riboflavin_y.csv");
    if (!fY.is_open()) throw runtime_error("Cannot open data/riboflavin_y.csv");
    getline(fY, line);
    vector<double> fullY;
    while (getline(fY, line)) {
        istringstream ss(line);
        string tok;
        bool firstToken = true;
        double val = 0; bool gotVal = false;
        while (getline(ss, tok, ',')) {
            if (firstToken) {
                firstToken = false;
                bool isNum = !tok.empty() &&
                             (isdigit((unsigned char)tok[0]) || tok[0]=='-' || tok[0]=='.');
                if (!isNum) continue;
            }
            try { val = stod(tok); gotVal = true; break; } catch (...) {}
        }
        if (gotVal) fullY.push_back(val);
    }
    fY.close();
    if ((int)fullY.size() != TOTAL_ROWS)
        throw runtime_error("riboflavin_y.csv: expected " + to_string(TOTAL_ROWS) +
                            " rows, got " + to_string(fullY.size()));

    // Column-normalize X (must match server)
    size_t nrows = fullX.size(), ncols = fullX[0].size();
    for (size_t j = 0; j < ncols; j++) {
        double nm = 0;
        for (size_t i = 0; i < nrows; i++) nm += fullX[i][j] * fullX[i][j];
        nm = sqrt(nm);
        if (nm > 1e-12)
            for (size_t i = 0; i < nrows; i++) fullX[i][j] /= nm;
    }

    // Center y
    double mu = 0;
    for (double v : fullY) mu += v;
    mu /= fullY.size();
    for (double& v : fullY) v -= mu;

    // Extract partition for this client
    int r0 = CLIENT_ROW_START[pid];
    int r1 = CLIENT_ROW_END[pid];
    Ai.clear(); bi.clear();
    for (int r = r0; r < r1; r++) {
        Ai.push_back(fullX[r]);
        bi.push_back(fullY[r]);
    }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) { cerr << "Usage: ./client_real <party_id 0|1|2> [--triad-only]\n"; return 1; }
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
    cout << "=== TRIAD Client (Riboflavin) " << myId << " ===" << endl;
    cout << "  N_FEAT=" << N_FEAT << " lambda=" << lambda_lasso
         << " rows=[" << CLIENT_ROW_START[myId] << "," << CLIENT_ROW_END[myId] << ")"
         << " Lmin=" << Lmin_levels << endl;

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
    // Precompute M_i, g_i from Riboflavin data
    // -----------------------------------------------------------------------
    cout << "\n[Precompute] Loading data and computing M_i, g_i..." << endl;
    Mat Ai; vector<double> bi;
    loadRiboflavinClient(myId, Ai, bi);
    size_t mi = Ai.size();  // number of rows for this client (24 or 23)
    cout << "  Data loaded: " << mi << " rows x " << Ai[0].size()
         << " cols (t=" << elapsed() << "s)" << endl;

    // x-update via Woodbury identity (avoids 4088x4088 inversion):
    //   (rho*I + A'A)^{-1} v = (1/rho)*v - (1/rho)*A'*(I_m + (1/rho)*A*A')^{-1}*(1/rho)*A*v
    // Only need to invert an m×m matrix (m=24 or 23).
    auto At  = transpose(Ai);                     // N_FEAT x mi
    auto AAt = matmul(Ai, At);                    // mi x mi
    // B = I_m + (1/rho)*A*A'
    Mat B = eye(mi, 1.0);
    for (size_t r = 0; r < mi; r++)
        for (size_t c = 0; c < mi; c++)
            B[r][c] += (1.0/rho) * AAt[r][c];
    auto Binv = invertSPD(B);                     // mi x mi  (fast!)
    auto gi   = matvec(At, bi);                   // N_FEAT
    cout << "  Woodbury precompute done (t=" << elapsed() << "s)" << endl;

    // Helper: x_i = M_i * (g_i + rho*v_i)  using Woodbury
    // = (1/rho)*(g_i+rho*v_i) - (1/rho)*A'*Binv*(1/rho)*A*(g_i+rho*v_i)
    auto woodburyXupdate = [&](const vector<double>& v) -> vector<double> {
        // rhs = g_i + rho*v_i
        vector<double> rhs(N_FEAT);
        for (size_t j = 0; j < N_FEAT; j++) rhs[j] = gi[j] + rho * v[j];
        // term1 = (1/rho)*rhs
        vector<double> term1(N_FEAT);
        for (size_t j = 0; j < N_FEAT; j++) term1[j] = rhs[j] / rho;
        // Arhs = A * rhs  (mi-vector)
        vector<double> Arhs(mi, 0.0);
        for (size_t r = 0; r < mi; r++)
            for (size_t j = 0; j < N_FEAT; j++) Arhs[r] += Ai[r][j] * rhs[j];
        // BinvArhs = Binv * Arhs  (mi-vector)
        vector<double> BinvArhs(mi, 0.0);
        for (size_t r = 0; r < mi; r++)
            for (size_t c = 0; c < mi; c++) BinvArhs[r] += Binv[r][c] * Arhs[c];
        // correction = (1/rho)*A'*BinvArhs  (N_FEAT-vector)
        vector<double> corr(N_FEAT, 0.0);
        for (size_t j = 0; j < N_FEAT; j++)
            for (size_t r = 0; r < mi; r++) corr[j] += At[j][r] * BinvArhs[r];
        // result = term1 - (1/rho)*corr
        vector<double> result(N_FEAT);
        for (size_t j = 0; j < N_FEAT; j++) result[j] = term1[j] - corr[j] / rho;
        return result;
    };

    // -----------------------------------------------------------------------
    // Distributed Plaintext ADMM
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
            auto vi_plain = recvVec(fd);
            auto xi_plain = woodburyXupdate(vi_plain);
            sendVec(fd, xi_plain);
        }

        // After PlainADMM: expect Static-R=2.0 phase
        uint32_t sig2 = recvU32(fd);
        if (sig2 != MAGIC_STATIC_R)
            throw runtime_error("Expected MAGIC_STATIC_R after PlainADMM, got: " + to_string(sig2));
        double sr = recvD(fd);
        cout << "[PlainADMM] Done. Starting Static-R=" << sr << "..." << endl;
    }
    } // end if (!triadOnly) — PlainADMM

    // -----------------------------------------------------------------------
    // Static-R = 2.0 experiment (between PlainADMM and TRIAD)
    // -----------------------------------------------------------------------
    if (!triadOnly) {
    {
        cout << "\n[Static-R] Running Static-R experiment..." << endl;
        vector<double> uiS(N_FEAT, 0.0);
        auto encUiS = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(vector<double>(N_FEAT, 0.0)));
        Ciphertext<DCRTPoly> encXiS_curr = encUiS;

        for (int iter = 0; iter < maxIter; iter++) {
            // Pre-iteration handshake: MAGIC_ABORT_SR or MAGIC_CONTINUE_ITER
            uint32_t hsig = recvU32(fd);
            if (hsig == MAGIC_ABORT_SR) {
                cout << "  [ABORT_SR] iter=" << iter << endl;
                break;
            }
            if (hsig != MAGIC_CONTINUE_ITER)
                throw runtime_error("Static-R: unexpected handshake " + to_string(hsig));

            cout << "  [Static-R] iter=" << setw(2) << iter
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

            // b) Participate in all v masked-decrypts; recover own v_i
            vector<double> viS(N_FEAT, 0.0);
            for (int j = 0; j < NUM_PARTIES; j++) {
                auto encVjMasked = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
                Ciphertext<DCRTPoly> partV;
                if (myId == 0) partV = cc->MultipartyDecryptLead({encVjMasked}, mySK)[0];
                else           partV = cc->MultipartyDecryptMain({encVjMasked}, mySK)[0];
                sendObj(fd, partV, cc);
                if (j == myId) {
                    auto viMasked = recvVec(fd);
                    for (size_t k = 0; k < N_FEAT; k++) viS[k] = viMasked[k] - rvS[k];
                }
            }

            // c) x-update and send enc(x_i)
            auto xiS = woodburyXupdate(viS);
            encXiS_curr = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(xiS));
            sendObj(fd, encXiS_curr, cc);

            // d) Participate in all u masked-decrypts; recover own u_i
            for (int j = 0; j < NUM_PARTIES; j++) {
                auto encUjMasked = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
                Ciphertext<DCRTPoly> partU;
                if (myId == 0) partU = cc->MultipartyDecryptLead({encUjMasked}, mySK)[0];
                else           partU = cc->MultipartyDecryptMain({encUjMasked}, mySK)[0];
                sendObj(fd, partU, cc);
                if (j == myId) {
                    auto uiMasked = recvVec(fd);
                    for (size_t k = 0; k < N_FEAT; k++) uiS[k] = uiMasked[k] - ruS[k];
                }
            }

            // e) (No CRC for Static-R — skip)

            // f) Receive enc(z^new)
            auto encZnewS = recvObj<Ciphertext<DCRTPoly>>(fd, cc);

            // g) Homomorphic u-update
            encUiS = cc->EvalAdd(encUiS, cc->EvalSub(encXiS_curr, encZnewS));
            sendObj(fd, encUiS, cc);

            // h) Handle MAGIC_REFRESH or MAGIC_ITERDONE
            uint32_t flagS = recvU32(fd);
            if (flagS == MAGIC_REFRESH) {
                cout << "    [Static-R] REFRESH" << endl;
                auto encZrefS = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
                Ciphertext<DCRTPoly> partZS;
                if (myId == 0) partZS = cc->MultipartyDecryptLead({encZrefS}, mySK)[0];
                else           partZS = cc->MultipartyDecryptMain({encZrefS}, mySK)[0];
                sendObj(fd, partZS, cc);
                auto encZfreshS = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
                (void)encZfreshS;
                encUiS = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(uiS));
                sendObj(fd, encUiS, cc);
                cout << "    [Static-R] Refresh done" << endl;
            }

            // i) Side-decrypt for server objective monitoring
            {
                auto encZobjS = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
                Ciphertext<DCRTPoly> partObjS;
                if (myId == 0) partObjS = cc->MultipartyDecryptLead({encZobjS}, mySK)[0];
                else           partObjS = cc->MultipartyDecryptMain({encZobjS}, mySK)[0];
                sendObj(fd, partObjS, cc);
            }

            cout << "    [Static-R iter " << iter << " done] t="
                 << fixed << setprecision(1) << elapsed() << "s" << endl;
            cout.flush();
        }

        // Final decrypt for Static-R
        if (recvU32(fd) != MAGIC_END)
            throw runtime_error("Expected MAGIC_END after Static-R loop");
        auto encZfinalS = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
        Ciphertext<DCRTPoly> partFinalS;
        if (myId == 0) partFinalS = cc->MultipartyDecryptLead({encZfinalS}, mySK)[0];
        else           partFinalS = cc->MultipartyDecryptMain({encZfinalS}, mySK)[0];
        sendObj(fd, partFinalS, cc);
        auto zFinalS = recvVec(fd);
        zFinalS.resize(N_FEAT);
        cout << "[Static-R] Done (t=" << fixed << setprecision(1) << elapsed() << "s)" << endl;

        // Expect MAGIC_TRIAD_START
        uint32_t sig3 = recvU32(fd);
        if (sig3 != MAGIC_TRIAD_START)
            throw runtime_error("Expected MAGIC_TRIAD_START after Static-R, got: " + to_string(sig3));
        cout << "[Static-R] Proceeding to TRIAD..." << endl;
    }
    } // end if (!triadOnly) — Static-R
    else {
        uint32_t sig = recvU32(fd);
        if (sig != MAGIC_TRIAD_START)
            throw runtime_error("Expected MAGIC_TRIAD_START in triad-only mode");
        cout << "[TRIAD-only] Received MAGIC_TRIAD_START, proceeding to Phase 1..." << endl;
    }

    // -----------------------------------------------------------------------
    // Phase 1: Bootstrap R
    // -----------------------------------------------------------------------
    cout << "\n[Phase 1] Bootstrap R..." << endl;
    auto xi    = woodburyXupdate(vector<double>(N_FEAT, 0.0));
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

        // a) Send fresh masks
        vector<double> rv(N_FEAT), ru(N_FEAT);
        for (auto& v : rv) v = maskDist(rng);
        for (auto& v : ru) v = maskDist(rng);
        auto encRv = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(rv));
        auto encRu = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(ru));
        sendObj(fd, encRv, cc);
        sendObj(fd, encRu, cc);
        cout << "    [a] masks sent" << endl; cout.flush();

        // b) Phase A: participate in ALL v masked-decrypts
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
        xi = woodburyXupdate(vi);
        encXi_curr = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(xi));
        sendObj(fd, encXi_curr, cc);
        cout << "    [c] enc(x_i) sent" << endl; cout.flush();

        // d) Phase C: participate in ALL u masked-decrypts
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
            bool safe = (maxW <= delta_safe * currentR);
            sendBool(fd, safe);
            // Mirror server R update
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

        // g) Homomorphic u-update
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
            auto encZfresh = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
            (void)encZfresh;
            encUi = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(ui));
            sendObj(fd, encUi, cc);
            cout << "    [h] Refresh done" << endl;
        }

        // i) Side-decrypt for server objective monitoring
        {
            auto encZobj = recvObj<Ciphertext<DCRTPoly>>(fd, cc);
            Ciphertext<DCRTPoly> partObj;
            if (myId == 0) partObj = cc->MultipartyDecryptLead({encZobj}, mySK)[0];
            else           partObj = cc->MultipartyDecryptMain({encZobj}, mySK)[0];
            sendObj(fd, partObj, cc);
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

    ofstream out("client_real" + to_string(myId) + "_result.csv");
    for (size_t i = 0; i < N_FEAT; i++) out << i << "," << zFinal[i] << "\n";
    out.close();

    netClose(fd);
    return 0;
}
