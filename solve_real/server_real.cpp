/**
 * @file server_real.cpp
 * @brief TRIAD Protocol - Server, Riboflavin real-data experiment
 *
 * Runs PlainADMM and Adaptive TRIAD on the Riboflavin dataset (71×4088).
 * Does NOT include Static-R experiments (only PlainADMM vs TRIAD comparison).
 *
 * Data layout across 3 clients:
 *   client 0: rows  0-23  (24 samples)
 *   client 1: rows 24-47  (24 samples)
 *   client 2: rows 48-70  (23 samples)
 *
 * Data files (relative to working directory):
 *   data/riboflavin_X.csv   (71 rows × 4088 cols, with header)
 *   data/riboflavin_y.csv   (71 rows × 1 col,  with header)
 *
 * Output logs (all with _real suffix):
 *   plaintext_admm_real_log.csv
 *   Adaptive_TRIAD_real_log.csv
 *   timing_real_log.csv
 *
 * Usage:
 *   ./server_real [--triad-only]
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
// Config
// ============================================================================
static const int    NUM_PARTIES    = 3;
static const int    BASE_PORT      = 10000;
static const size_t N_FEAT         = 4088;
static const size_t M_ROWS_MAX     = 24;   // max rows per client (client 2 has 23)
static const int    TOTAL_ROWS     = 71;
static const double rho            = 1.0;
static const double lambda_lasso   = 0.01;  // smaller lambda for real data
static const int    maxIter        = 100;
static const int    updateInterval = 5;
static const int    shrinkWarmup   = 5;
static const int    chebyDegree    = 15;
static const double alpha_crc      = 1.2;
static const double gamma_smooth   = 0.8;
static const double delta_safe     = 0.95;
static const int    Lmin_levels    = 6;
static const double kappa          = lambda_lasso / (rho * NUM_PARTIES);

static const char* CLIENT_IPS[3] = {
    "192.168.186.223",
    "192.168.186.33",
    "192.168.186.214"
};

// rows assigned to each client
static const int CLIENT_ROW_START[3] = {0,  24, 48};
static const int CLIENT_ROW_END[3]   = {24, 48, 71};

static const uint32_t MAGIC_READY        = 0xCAFEBABE;
static const uint32_t MAGIC_REFRESH      = 0xABCD1234;
static const uint32_t MAGIC_ITERDONE     = 0x00000001;
static const uint32_t MAGIC_END          = 0xFFFFFFFF;
static const uint32_t MAGIC_PLAIN_ADMM   = 0xAAAAAAAA;
static const uint32_t MAGIC_TRIAD_START  = 0xBBBBBBBB;
static const uint32_t MAGIC_STATIC_R     = 0xCCCCCCCC;
static const uint32_t MAGIC_ALL_DONE     = 0xEEEEEEEE;
static const uint32_t MAGIC_ABORT_SR     = 0xCC1EAF01;
static const uint32_t MAGIC_CONTINUE_ITER= 0x00000003;

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
// Chebyshev coefficients for soft-threshold on [-R,R]
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
// maskedDecryptSend
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
// Data helpers
// ============================================================================
using Mat = vector<vector<double>>;

// Load full Riboflavin X and y (71 rows)
static void loadRiboflavinFull(Mat& X, vector<double>& y) {
    // Load X
    ifstream fX("data/riboflavin_X.csv");
    if (!fX.is_open()) throw runtime_error("Cannot open data/riboflavin_X.csv");
    string line;
    getline(fX, line);  // skip header
    X.clear();
    while (getline(fX, line)) {
        istringstream ss(line);
        string tok;
        vector<double> row;
        // skip row-name field if present (non-numeric first token)
        bool firstToken = true;
        while (getline(ss, tok, ',')) {
            if (firstToken) {
                firstToken = false;
                // check if it looks like a number
                bool isNum = !tok.empty() &&
                             (isdigit((unsigned char)tok[0]) || tok[0]=='-' || tok[0]=='.');
                if (!isNum) continue;  // skip row label
            }
            try { row.push_back(stod(tok)); } catch (...) {}
        }
        if (!row.empty()) X.push_back(row);
    }
    fX.close();
    if ((int)X.size() != TOTAL_ROWS)
        throw runtime_error("riboflavin_X.csv: expected " + to_string(TOTAL_ROWS) +
                            " rows, got " + to_string(X.size()));

    // Load y
    ifstream fY("data/riboflavin_y.csv");
    if (!fY.is_open()) throw runtime_error("Cannot open data/riboflavin_y.csv");
    getline(fY, line);  // skip header
    y.clear();
    while (getline(fY, line)) {
        istringstream ss(line);
        string tok;
        bool firstToken = true;
        double val = 0;
        bool gotVal = false;
        while (getline(ss, tok, ',')) {
            if (firstToken) {
                firstToken = false;
                bool isNum = !tok.empty() &&
                             (isdigit((unsigned char)tok[0]) || tok[0]=='-' || tok[0]=='.');
                if (!isNum) continue;
            }
            try { val = stod(tok); gotVal = true; break; } catch (...) {}
        }
        if (gotVal) y.push_back(val);
    }
    fY.close();
    if ((int)y.size() != TOTAL_ROWS)
        throw runtime_error("riboflavin_y.csv: expected " + to_string(TOTAL_ROWS) +
                            " rows, got " + to_string(y.size()));
}

// Column-normalize X in-place (each column divided by its L2 norm)
static void colNormalize(Mat& X) {
    size_t nrows = X.size(), ncols = X[0].size();
    for (size_t j = 0; j < ncols; j++) {
        double nm = 0;
        for (size_t i = 0; i < nrows; i++) nm += X[i][j] * X[i][j];
        nm = sqrt(nm);
        if (nm > 1e-12)
            for (size_t i = 0; i < nrows; i++) X[i][j] /= nm;
    }
}

// Center y (subtract mean)
static void centerY(vector<double>& y) {
    double mu = 0;
    for (double v : y) mu += v;
    mu /= y.size();
    for (double& v : y) v -= mu;
}

// Load global (all 71 rows) after normalization
static void loadGlobal(Mat& globalA, vector<double>& globalB) {
    loadRiboflavinFull(globalA, globalB);
    colNormalize(globalA);
    centerY(globalB);
}

// Load partition for client pid (server uses same logic for generating v_i etc)
static void loadPartition(int pid, Mat& Ai, vector<double>& bi,
                           const Mat& globalA, const vector<double>& globalB) {
    int r0 = CLIENT_ROW_START[pid];
    int r1 = CLIENT_ROW_END[pid];
    Ai.clear(); bi.clear();
    for (int r = r0; r < r1; r++) {
        Ai.push_back(globalA[r]);
        bi.push_back(globalB[r]);
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

static double computeMSE(const vector<double>& z,
                          const Mat& A, const vector<double>& b) {
    double mse = 0.0;
    size_t n = A.size();
    for (size_t i = 0; i < n; i++) {
        double Az = 0.0;
        for (size_t j = 0; j < z.size(); j++) Az += A[i][j] * z[j];
        double d = Az - b[i]; mse += d * d;
    }
    return mse / n;
}

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
    cout << "=== TRIAD Server (Riboflavin, K=" << NUM_PARTIES << ")"
         << (triadOnly ? " [TRIAD-ONLY]" : "") << " ===" << endl;
    cout << "  N_FEAT=" << N_FEAT << " TOTAL_ROWS=" << TOTAL_ROWS
         << " lambda=" << lambda_lasso << " kappa=" << kappa
         << " Lmin=" << Lmin_levels << endl;

    // Load global dataset
    cout << "\n[Data] Loading Riboflavin dataset..." << endl;
    Mat globalA; vector<double> globalB;
    loadGlobal(globalA, globalB);
    cout << "  Loaded " << globalA.size() << " rows x " << globalA[0].size()
         << " cols, y size=" << globalB.size() << " (t=" << elapsed() << "s)" << endl;

    // Load per-client partitions (server needs them for PlainADMM and objective)
    vector<Mat>           clientA(NUM_PARTIES);
    vector<vector<double>> clientB(NUM_PARTIES);
    for (int i = 0; i < NUM_PARTIES; i++)
        loadPartition(i, clientA[i], clientB[i], globalA, globalB);

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

    // Verify BatchSize is sufficient
    if (cc->GetEncodingParams()->GetBatchSize() < N_FEAT)
        throw runtime_error("BatchSize " +
            to_string(cc->GetEncodingParams()->GetBatchSize()) +
            " < N_FEAT " + to_string(N_FEAT) +
            ". Regenerate keys with keygen_real.");

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
    vector<double> zPlain(N_FEAT, 0.0);
    double plainFinalObj = 0.0;

    if (!triadOnly) {
    cout << "\n[Plaintext ADMM] Starting distributed baseline..." << endl;
    bcastU32(fds, MAGIC_PLAIN_ADMM);

    {
        vector<double> zP(N_FEAT, 0.0);
        vector<vector<double>> uP(NUM_PARTIES, vector<double>(N_FEAT, 0.0));

        ofstream plog("plaintext_admm_real_log.csv");
        plog << "iter,objective,mse,elapsed_s\n";

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
            double mse = computeMSE(zP, globalA, globalB);
            cout << "  [PlainADMM] iter=" << setw(2) << iter
                 << "  max|w|=" << fixed << setprecision(4) << maxW
                 << "  obj=" << fixed << setprecision(6) << obj
                 << "  mse=" << fixed << setprecision(6) << mse
                 << "  t=" << fixed << setprecision(2) << elapsed() << "s" << endl;
            plog << iter << "," << obj << "," << mse << "," << elapsed() << "\n";
            plog.flush();
        }
        plog.close();
        zPlain = zP;
        plainFinalObj = computeObjective(zPlain, globalA, globalB);
        cout << "[Plaintext ADMM] Done. Final obj=" << plainFinalObj << endl;
    }
    } // end if (!triadOnly)

    // -----------------------------------------------------------------------
    // Static-R = 2.0 experiment (for comparison with TRIAD)
    // -----------------------------------------------------------------------
    double staticR2FinalObj = std::numeric_limits<double>::quiet_NaN();
    double staticR2FinalMSE = std::numeric_limits<double>::quiet_NaN();
    if (!triadOnly) {
    {
        const double sr = 2.0;
        cout << "\n===== Static-R=" << sr << " =====" << endl;
        bcastU32(fds, MAGIC_STATIC_R);
        bcastD(fds, sr);

        auto ptZ0 = cc->MakeCKKSPackedPlaintext(vector<double>(N_FEAT, 0.0));
        auto encZS = cc->Encrypt(jointPK, ptZ0);
        vector<Ciphertext<DCRTPoly>> encUS(NUM_PARTIES);
        for (int i = 0; i < NUM_PARTIES; i++)
            encUS[i] = cc->Encrypt(jointPK, ptZ0);
        auto coeffS = chebyCoeffs(sr, chebyDegree);

        ofstream slog("StaticR_2_0_real_log.csv");
        slog << "iter,R,remLev,refresh,objective,mse,elapsed_s\n";
        bool exploded   = false;
        bool abort_next = false;
        double prev_obj = -1.0;

        for (int iter = 0; iter < maxIter; iter++) {
            bool did_ref = false;
            size_t remLev = totalLevels - encZS->GetLevel() - 1;

            if (abort_next) {
                cout << "  [ABORT] StaticR=2.0 aborted at iter=" << iter << endl;
                bcastU32(fds, MAGIC_ABORT_SR);
                slog << iter << "," << sr << "," << remLev
                     << "," << 0 << ",ABORT,ABORT," << elapsed() << "\n";
                slog.flush();
                break;
            }
            bcastU32(fds, MAGIC_CONTINUE_ITER);

            cout << "  iter=" << setw(2) << iter
                 << "  R=" << sr << "  remLev=" << remLev
                 << "  t=" << fixed << setprecision(1) << elapsed() << "s" << endl;

            // a) recv masks
            vector<Ciphertext<DCRTPoly>> encRvS(NUM_PARTIES), encRuS(NUM_PARTIES);
            for (int i = 0; i < NUM_PARTIES; i++) {
                encRvS[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
                encRuS[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
            }

            // b) v masked-decrypts
            for (int i = 0; i < NUM_PARTIES; i++)
                maskedDecryptSend(i, fds, cc->EvalSub(encZS, encUS[i]), encRvS[i], cc, N_FEAT);

            // c) collect enc(x_i)
            vector<Ciphertext<DCRTPoly>> encXS(NUM_PARTIES);
            for (int i = 0; i < NUM_PARTIES; i++)
                encXS[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);

            // d) u masked-decrypts
            for (int i = 0; i < NUM_PARTIES; i++)
                maskedDecryptSend(i, fds, encUS[i], encRuS[i], cc, N_FEAT);

            // e) enc(w)
            auto encWS = cc->EvalAdd(encXS[0], encUS[0]);
            for (int i = 1; i < NUM_PARTIES; i++)
                encWS = cc->EvalAdd(encWS, cc->EvalAdd(encXS[i], encUS[i]));
            encWS = cc->EvalMult(encWS, 1.0 / NUM_PARTIES);

            // f) Chebyshev z-update with fixed R
            auto encZnewS = cc->EvalChebyshevSeries(
                                cc->EvalMult(encWS, 1.0 / sr), coeffS, -1.0, 1.0);

            // g) broadcast enc(z), collect enc(u_i)
            bcast(fds, encZnewS, cc);
            for (int i = 0; i < NUM_PARTIES; i++)
                encUS[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
            encZS = encZnewS;

            // h) Refresh or ITERDONE
            remLev = totalLevels - encZS->GetLevel() - 1;
            if ((int)remLev < Lmin_levels) {
                did_ref = true;
                bcastU32(fds, MAGIC_REFRESH);
                bcast(fds, encZS, cc);
                vector<Ciphertext<DCRTPoly>> zshS(NUM_PARTIES);
                for (int i = 0; i < NUM_PARTIES; i++)
                    zshS[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
                Plaintext ptZpS; cc->MultipartyDecryptFusion(zshS, &ptZpS); ptZpS->SetLength(N_FEAT);
                encZS = cc->Encrypt(jointPK, cc->MakeCKKSPackedPlaintext(ptZpS->GetRealPackedValue()));
                bcast(fds, encZS, cc);
                for (int i = 0; i < NUM_PARTIES; i++)
                    encUS[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
            } else {
                bcastU32(fds, MAGIC_ITERDONE);
            }

            // i) side-decrypt for objective monitoring
            double obj = -1.0, mse_s = -1.0;
            {
                bcast(fds, encZS, cc);
                vector<Ciphertext<DCRTPoly>> objShS(NUM_PARTIES);
                for (int i = 0; i < NUM_PARTIES; i++)
                    objShS[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
                try {
                    Plaintext ptObjS; cc->MultipartyDecryptFusion(objShS, &ptObjS);
                    ptObjS->SetLength(N_FEAT);
                    auto zvecS = ptObjS->GetRealPackedValue(); zvecS.resize(N_FEAT);
                    obj   = computeObjective(zvecS, globalA, globalB);
                    mse_s = computeMSE(zvecS, globalA, globalB);
                    bool burst   = (prev_obj > 0.0 && obj > prev_obj * 100.0 && obj > 100.0);
                    bool abs_exp = (obj > 1e4);
                    bool nan_inf = (std::isnan(obj) || std::isinf(obj));
                    if (!exploded && (burst || abs_exp || nan_inf)) {
                        string reason = nan_inf ? "NaN/Inf" : (burst ? "100x_jump" : "obj>1e4");
                        cout << "    [EXPLOSION] obj=" << obj << " reason=" << reason << endl;
                        exploded = true; abort_next = true;
                    } else {
                        cout << "    obj=" << fixed << setprecision(6) << obj
                             << "  mse=" << fixed << setprecision(6) << mse_s << endl;
                        prev_obj = obj;
                    }
                } catch (...) {
                    cout << "    [DECRYPT_FAIL] iter=" << iter << endl;
                    abort_next = true; exploded = true;
                }
            }
            slog << iter << "," << sr << "," << (totalLevels-encZS->GetLevel()-1)
                 << "," << did_ref << ","
                 << (obj<0?"NaN":to_string(obj)) << ","
                 << (mse_s<0?"NaN":to_string(mse_s)) << ","
                 << elapsed() << "\n";
            slog.flush();
        }
        slog.close();

        // Final decrypt for Static-R=2.0
        bcastU32(fds, MAGIC_END);
        bcast(fds, encZS, cc);
        vector<Ciphertext<DCRTPoly>> fshS(NUM_PARTIES);
        for (int i = 0; i < NUM_PARTIES; i++)
            fshS[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
        try {
            Plaintext ptFS; cc->MultipartyDecryptFusion(fshS, &ptFS);
            ptFS->SetLength(N_FEAT);
            auto zFS = ptFS->GetRealPackedValue(); zFS.resize(N_FEAT);
            for (int i = 0; i < NUM_PARTIES; i++) sendVec(fds[i], zFS);
            staticR2FinalObj = computeObjective(zFS, globalA, globalB);
            staticR2FinalMSE = computeMSE(zFS, globalA, globalB);
            cout << "  StaticR=2.0 final obj=" << staticR2FinalObj
                 << " mse=" << staticR2FinalMSE << endl;
        } catch (...) {
            cout << "  [FINAL_DECRYPT_FAIL] StaticR=2.0" << endl;
            vector<double> zeroVec(N_FEAT, 0.0);
            for (int i = 0; i < NUM_PARTIES; i++) sendVec(fds[i], zeroVec);
        }
    }
    } // end if (!triadOnly) — Static-R=2.0

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
    cout << "\n[Phase 2] Running " << maxIter << " iterations..." << endl;

    ofstream log("Adaptive_TRIAD_real_log.csv");
    log << "iter,R,remLev,crc,refresh,objective,mse,elapsed_s\n";
    ofstream tlog("timing_real_log.csv");
    tlog << "iter,t_mask_ms,t_vdecrypt_ms,t_xcollect_ms,t_udecrypt_ms,t_wupdate_ms"
            ",t_crc_ms,t_cheby_ms,t_broadcast_ms,t_refresh_ms,t_sidedecrypt_ms,t_total_ms\n";

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

    for (int iter = 0; iter < maxIter; iter++) {
        auto tp_start = hrc::now();
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
        auto tp_a = hrc::now();

        // b) masked decrypt ALL v_i
        for (int i = 0; i < NUM_PARTIES; i++)
            maskedDecryptSend(i, fds, cc->EvalSub(encZ, encU[i]), encRv[i], cc, N_FEAT);
        auto tp_b = hrc::now();

        // c) collect enc(x_i)
        vector<Ciphertext<DCRTPoly>> encX(NUM_PARTIES);
        for (int i = 0; i < NUM_PARTIES; i++)
            encX[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
        auto tp_c = hrc::now();

        // d) masked decrypt ALL u_i
        for (int i = 0; i < NUM_PARTIES; i++)
            maskedDecryptSend(i, fds, encU[i], encRu[i], cc, N_FEAT);
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

        // j) side-decrypt for objective monitoring
        double obj = -1.0, mse_t = -1.0;
        {
            bcast(fds, encZ, cc);
            vector<Ciphertext<DCRTPoly>> objSh(NUM_PARTIES);
            for (int i = 0; i < NUM_PARTIES; i++)
                objSh[i] = recvObj<Ciphertext<DCRTPoly>>(fds[i], cc);
            try {
                Plaintext ptObj; cc->MultipartyDecryptFusion(objSh, &ptObj);
                ptObj->SetLength(N_FEAT);
                auto zvec = ptObj->GetRealPackedValue(); zvec.resize(N_FEAT);
                obj   = computeObjective(zvec, globalA, globalB);
                mse_t = computeMSE(zvec, globalA, globalB);
                cout << "    obj=" << fixed << setprecision(6) << obj
                     << "  mse=" << fixed << setprecision(6) << mse_t << endl;
            } catch (...) { cout << "    obj=N/A" << endl; }
        }
        auto tp_j = hrc::now();

        double dt_mask       = hms(tp_start, tp_a);
        double dt_vdecrypt   = hms(tp_a,     tp_b);
        double dt_xcollect   = hms(tp_b,     tp_c);
        double dt_udecrypt   = hms(tp_c,     tp_d);
        double dt_wupdate    = hms(tp_d,     tp_e);
        double dt_crc        = hms(tp_e,     tp_f);
        double dt_cheby      = hms(tp_f,     tp_g);
        double dt_broadcast  = hms(tp_g,     tp_h);
        double dt_refresh    = hms(tp_h,     tp_i);
        double dt_sidedecrypt= hms(tp_i,     tp_j);
        double dt_total      = hms(tp_start, tp_j);

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
            << did_crc << "," << did_ref << ","
            << (obj<0?"NaN":to_string(obj)) << ","
            << (mse_t<0?"NaN":to_string(mse_t)) << "," << elapsed() << "\n";
        log.flush();
        tlog << iter << fixed << setprecision(3)
             << "," << dt_mask << "," << dt_vdecrypt << "," << dt_xcollect
             << "," << dt_udecrypt << "," << dt_wupdate << "," << dt_crc
             << "," << dt_cheby << "," << dt_broadcast << "," << dt_refresh
             << "," << dt_sidedecrypt << "," << dt_total << "\n";
        tlog.flush();
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

    double triadFinalObj = computeObjective(zF, globalA, globalB);
    cout << "Final objective (TRIAD): " << triadFinalObj << endl;
    cout << "Total time: " << elapsed() << "s" << endl;

    // -----------------------------------------------------------------------
    // Results summary
    // -----------------------------------------------------------------------
    cout << "\n============================================================" << endl;
    cout << " Objective Comparison: PlainADMM vs Adaptive TRIAD (Riboflavin)" << endl;
    cout << "============================================================" << endl;
    cout << left << setw(26) << "Algorithm"
         << right << setw(18) << "Final Objective" << endl;
    cout << string(46, '-') << endl;
    if (!triadOnly)
        cout << left << setw(26) << "Plaintext_ADMM"
             << right << setw(18) << fixed << setprecision(6) << plainFinalObj << "\n";
    cout << left << setw(26) << "Adaptive_TRIAD"
         << right << setw(18) << fixed << setprecision(6) << triadFinalObj << "\n";
    cout << "============================================================" << endl;

    // Latency breakdown
    {
        int N = (int)v_t_total.size();
        if (N > 0) {
            auto mean_v = [&](const vector<double>& v) {
                double s = 0; for (double x : v) s += x; return s / v.size();
            };
            double mu_total = mean_v(v_t_total);
            cout << "\n============================================================" << endl;
            cout << " Per-Iteration Latency (Adaptive TRIAD, Riboflavin)" << endl;
            cout << "  (" << N << " iterations, times in ms)" << endl;
            cout << "============================================================" << endl;
            auto row = [&](const char* name, double mu) {
                cout << left << setw(24) << name
                     << right << setw(10) << fixed << setprecision(2) << mu << " ms\n";
            };
            row("(a) recv_masks",       mean_v(v_t_mask));
            row("(b) v_masked_decrypt", mean_v(v_t_vdecrypt));
            row("(c) x_collect",        mean_v(v_t_xcollect));
            row("(d) u_masked_decrypt", mean_v(v_t_udecrypt));
            row("(e) w_update",         mean_v(v_t_wupdate));
            row("(f) CRC",              mean_v(v_t_crc));
            row("(g) Chebyshev",        mean_v(v_t_cheby));
            row("(h) broadcast_z+u",    mean_v(v_t_broadcast));
            row("(i) refresh",          mean_v(v_t_refresh));
            row("(j) side_decrypt",     mean_v(v_t_sidedecrypt));
            cout << string(36, '-') << endl;
            row("TOTAL",                mu_total);
            cout << "============================================================" << endl;
        }
    }

    // R adaptation history
    {
        cout << "\n============================================================" << endl;
        cout << " R Adaptation History (Adaptive TRIAD, Riboflavin)" << endl;
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
                     << setw(10) << fixed << setprecision(4) << e.newR
                     << setw(10) << "-" << setw(10) << fixed << setprecision(4) << e.newR
                     << setw(12) << "-" << setw(10) << "-" << setw(7) << "-" << "\n";
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
