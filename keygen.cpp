/**
 * @file keygen.cpp
 * @brief TRIAD Trusted Dealer - Threshold CKKS Key Generation
 *
 * Follows OpenFHE's official threshold-fhe.cpp CKKS pattern EXACTLY.
 * No joint secret key is saved — true threshold scheme.
 *
 * EvalSum/Rot chaining pattern (from threshold-fhe-5p.cpp):
 *   party 0: EvalSumKeyGen(sk0)          -> sum1 (tag=kp0.pk.tag)
 *   party 1: MultiEvalSumKeyGen(sk1, sum1, kp1.pk.tag)  -> sum2
 *            MultiAddEvalSumKeys(sum1, sum2, kp1.pk.tag) -> sum12
 *   party 2: MultiEvalSumKeyGen(sk2, sum12, kp2.pk.tag) -> sum3
 *            MultiAddEvalSumKeys(sum12, sum3, kp2.pk.tag)-> sum123  <- insert
 *
 * Parameters: same as step4 (MultDepth=30, ScalingModSize=40, BatchSize=256)
 */

#include "openfhe/pke/openfhe.h"
#include "openfhe/pke/scheme/ckksrns/ckksrns-ser.h"
#include "openfhe/pke/key/key-ser.h"
#include "openfhe/pke/cryptocontext-ser.h"
#include <iostream>
#include <filesystem>
#include <chrono>

using namespace lbcrypto;
using namespace std;

static const int NUM_PARTIES      = 3;
static const int MULT_DEPTH       = 15;   // sufficient with Refresh; forces RingDim=32768
static const int SCALING_MOD_SIZE = 40;
static const int FIRST_MOD_SIZE   = 45;
static const int BATCH_SIZE       = 256;

int main() {
    auto t0 = chrono::high_resolution_clock::now();
    auto elapsed = [&]() {
        return chrono::duration<double>(
            chrono::high_resolution_clock::now() - t0).count();
    };

    cout << "=== TRIAD Trusted Dealer Key Generation ===" << endl;
    filesystem::create_directories("keys");

    // 1. CryptoContext
    CCParams<CryptoContextCKKSRNS> params;
    params.SetSecretKeyDist(UNIFORM_TERNARY);
    params.SetSecurityLevel(HEStd_NotSet);  // disable auto ring-upgrade
    params.SetRingDim(32768);               // force 32768 for speed
    params.SetMultiplicativeDepth(MULT_DEPTH);
    params.SetScalingModSize(SCALING_MOD_SIZE);
    params.SetFirstModSize(FIRST_MOD_SIZE);
    params.SetBatchSize(BATCH_SIZE);
    params.SetKeySwitchTechnique(HYBRID);
    params.SetScalingTechnique(FLEXIBLEAUTO);

    auto cc = GenCryptoContext(params);
    cc->Enable(PKE); cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE); cc->Enable(ADVANCEDSHE); cc->Enable(MULTIPARTY);

    cout << "RingDim=" << cc->GetRingDimension()
         << " BatchSize=" << cc->GetEncodingParams()->GetBatchSize()
         << " (t=" << elapsed() << "s)" << endl;
    Serial::SerializeToFile("keys/cryptocontext.bin", cc, SerType::BINARY);

    // 2. Individual keypairs: kp[0]=KeyGen, kp[i]=MultipartyKeyGen(kp[i-1].pk)
    vector<KeyPair<DCRTPoly>> kp(NUM_PARTIES);
    kp[0] = cc->KeyGen();
    for (int i = 1; i < NUM_PARTIES; i++)
        kp[i] = cc->MultipartyKeyGen(kp[i-1].publicKey);

    // Joint public key = last chained public key
    auto jointPK   = kp[NUM_PARTIES-1].publicKey;
    string jointTag = jointPK->GetKeyTag();

    for (int i = 0; i < NUM_PARTIES; i++)
        Serial::SerializeToFile("keys/sk_" + to_string(i) + ".bin",
                                kp[i].secretKey, SerType::BINARY);
    Serial::SerializeToFile("keys/joint_pk.bin", jointPK, SerType::BINARY);
    cout << "Keys saved, jointTag=" << jointTag << " (t=" << elapsed() << "s)" << endl;

    // 3. Threshold EvalMult (from threshold-fhe.cpp RunCKKS)
    auto em1   = cc->KeySwitchGen(kp[0].secretKey, kp[0].secretKey);
    auto em2   = cc->MultiKeySwitchGen(kp[1].secretKey, kp[1].secretKey, em1);
    auto em12  = cc->MultiAddEvalKeys(em1, em2, kp[1].publicKey->GetKeyTag());
    auto em3   = cc->MultiKeySwitchGen(kp[2].secretKey, kp[2].secretKey, em12);
    auto em123 = cc->MultiAddEvalKeys(em12, em3, jointTag);
    auto emM1  = cc->MultiMultEvalKey(kp[0].secretKey, em123, jointTag);
    auto emM2  = cc->MultiMultEvalKey(kp[1].secretKey, em123, jointTag);
    auto emM3  = cc->MultiMultEvalKey(kp[2].secretKey, em123, jointTag);
    auto emM123= cc->MultiAddEvalMultKeys(
                     cc->MultiAddEvalMultKeys(emM1, emM2, emM1->GetKeyTag()),
                     emM3, emM1->GetKeyTag());
    cc->InsertEvalMultKey({emM123});
    cout << "EvalMult done (t=" << elapsed() << "s)" << endl;

    // 4. Threshold EvalSum (from threshold-fhe-5p.cpp)
    cc->EvalSumKeyGen(kp[0].secretKey);
    auto sum1  = make_shared<map<usint,EvalKey<DCRTPoly>>>(
                     cc->GetEvalSumKeyMap(kp[0].publicKey->GetKeyTag()));
    auto sum2  = cc->MultiEvalSumKeyGen(kp[1].secretKey, sum1,
                                         kp[1].publicKey->GetKeyTag());
    auto sum12 = cc->MultiAddEvalSumKeys(sum1, sum2,
                                          kp[1].publicKey->GetKeyTag());
    auto sum3  = cc->MultiEvalSumKeyGen(kp[2].secretKey, sum12, jointTag);
    auto sum123= cc->MultiAddEvalSumKeys(sum12, sum3, jointTag);
    cc->InsertEvalSumKey(sum123);
    cout << "EvalSum done (t=" << elapsed() << "s)" << endl;

    // 5. Threshold EvalRotate (same chaining pattern as EvalSum)
    vector<int32_t> rotIdx;
    for (int i = 1; i < BATCH_SIZE; i *= 2) {
        rotIdx.push_back(i);
        rotIdx.push_back(-i);
    }
    cc->EvalAtIndexKeyGen(kp[0].secretKey, rotIdx);
    auto rot1  = make_shared<map<usint,EvalKey<DCRTPoly>>>(
                     cc->GetEvalAutomorphismKeyMap(kp[0].publicKey->GetKeyTag()));
    auto rot2  = cc->MultiEvalAtIndexKeyGen(kp[1].secretKey, rot1, rotIdx,
                                             kp[1].publicKey->GetKeyTag());
    auto rot12 = cc->MultiAddEvalAutomorphismKeys(rot1, rot2,
                                                   kp[1].publicKey->GetKeyTag());
    auto rot3  = cc->MultiEvalAtIndexKeyGen(kp[2].secretKey, rot12, rotIdx, jointTag);
    auto rot123= cc->MultiAddEvalAutomorphismKeys(rot12, rot3, jointTag);
    cc->InsertEvalAutomorphismKey(rot123);
    cout << "EvalRotate done (t=" << elapsed() << "s)" << endl;

    // 6. Save eval keys
    {
        const auto& v = cc->GetEvalMultKeyVector(emM123->GetKeyTag());
        Serial::SerializeToFile("keys/eval_mult_key.bin", v, SerType::BINARY);
    }
    Serial::SerializeToFile("keys/eval_sum_key.bin",  *sum123,  SerType::BINARY);
    Serial::SerializeToFile("keys/eval_rot_key.bin",  *rot123,  SerType::BINARY);

    // 7. Sizes
    vector<string> files = {
        "keys/cryptocontext.bin","keys/joint_pk.bin",
        "keys/eval_mult_key.bin","keys/eval_sum_key.bin","keys/eval_rot_key.bin"
    };
    for (int i = 0; i < NUM_PARTIES; i++)
        files.push_back("keys/sk_" + to_string(i) + ".bin");
    double total = 0;
    for (auto& f : files) {
        double mb = (double)filesystem::file_size(f) / 1048576.0;
        total += mb;
        cout << "  " << f << ": " << mb << " MB" << endl;
    }
    cout << "Total: " << total << " MB" << endl;
    cout << "\nfor i in 0 1 2; do" << endl;
    cout << "  scp keys/cryptocontext.bin keys/joint_pk.bin keys/sk_${i}.bin pi${i}:~/triad/keys/" << endl;
    cout << "done" << endl;
    cout << "Done in " << elapsed() << "s" << endl;
    return 0;
}