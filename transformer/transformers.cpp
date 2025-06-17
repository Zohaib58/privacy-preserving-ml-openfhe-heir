#define PROFILE
#include "openfhe/pke/openfhe.h"
#include <cmath>
#include <array>
using namespace lbcrypto;
using namespace std;
using Vector = vector<double>; 
using EmbeddingMatrix = vector<Vector>;
using DotProdMatrix = vector<vector<Ciphertext<DCRTPoly>>>;


// Adds sinusoidal positional encoding to token embeddings
EmbeddingMatrix addPositionalEncoding(const EmbeddingMatrix& embeddings){
    EmbeddingMatrix peMatrix;
    size_t words = embeddings.size();
    size_t dim = embeddings[0].size();

    for (int i = 0; i < words; i++){
        vector<double> wordPE = {};
        for (int j = 0; j < dim; j++)
        {
            if (j % 2 == 0)
            {
                wordPE.push_back(sin((i) / pow((10000), (j/dim))));

            }
            else{
                wordPE.push_back(cos((i) / pow((10000), (j/dim))));
            }
        }
        peMatrix.push_back(wordPE);
    }

    for (size_t i = 0; i < words; i++) {
        for (size_t j = 0; j < dim; j++) {
            peMatrix[i][j] += embeddings[i][j];
        }
    }

    return peMatrix;
}


vector<double> flattenMatrix(EmbeddingMatrix matrix){
    vector<double> flatVec;
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    
    for (size_t i = 0; i < rows; i++){
        for (size_t j = 0; j < cols; j++){
            flatVec.push_back(matrix[i][j]);
        }
    }

    return flatVec;
}

Plaintext MakeCKKSPackedTokens(vector<double> flattenPE, CryptoContext<DCRTPoly> cc){
    return cc->MakeCKKSPackedPlaintext(flattenPE);
}

/*
for each diagNum
A[0, 1, 2 % rowCountA, (diagNum + 0,1,2) % colCountA]

*/
Ciphertext<DCRTPoly> calculateUpperLowerDiagonal(Ciphertext<DCRTPoly> ctxt, int diagCount, int rowSize, int colCount, CryptoContext<DCRTPoly> cc, bool isLower){
    Ciphertext<DCRTPoly> result;
    bool first = true;
    
    for (int i = 0; i < diagCount; i++){
        vector<double> mask(12, 0.0);
            
        for (int j = 0; j < rowSize; j++){
            if (isLower)
                mask[((i + j) % rowSize) * colCount + (j % colCount)] = 1;
            else
                mask[(j % rowSize) * colCount + ((i + j) % colCount)] = 1;
             
        }

        auto calc = cc -> EvalMult(ctxt, cc -> MakeCKKSPackedPlaintext(mask));
        if (first) {
            result = calc;
            first = false;
        }
        else {
            result = cc -> EvalAdd(result, calc);
        }
    }
    return result;


}

/*
    for each diagNu
B[diagNum + 0, 1, 2 % rowCountB, (0,1,2) % colCountB]

*/



vector<double> calculateDiagonal(const EmbeddingMatrix& W, int diagNum) {
    size_t words = W.size();          // number of rows
    size_t dim = W[0].size();         // number of columns
    size_t total_slots = words * dim;

    vector<double> diag(total_slots, 0.0);

    for (size_t row = 0; row < words; row++) {
        for (size_t col = 0; col < dim; col++) {
            size_t original_index = row * dim + col;
            size_t rotated_index = (original_index + diagNum) % total_slots;
            diag[rotated_index] = W[row][col];
        }
    }

    return diag;
}



// Applies diagonal encoding based matrix-vector multiplication
Ciphertext<DCRTPoly> applyDiagonalProjection(const Ciphertext<DCRTPoly>& encPE,
                                                       const EmbeddingMatrix& W_,
                                                       CryptoContext<DCRTPoly> cc) {
    const size_t words = W_.size();
    const size_t dim = W_[0].size();
    
    vector<Plaintext> diagMat;
    for (size_t h = 0; h < 12; h++) {
        diagMat.push_back(cc->MakeCKKSPackedPlaintext(calculateDiagonal(W_, h)));
    }
    
    Ciphertext<DCRTPoly> p;
    for (int j = 0; j < 12; j++){
        auto product = cc -> EvalMult(cc -> EvalRotate(encPE, j), diagMat[j]); 
        p = (j==0) ? product : cc -> EvalAdd(p, product);
    }
 
    return p;
}

vector<double> generateMask(bool isUpper, size_t diagNum, size_t dim){
    vector<double> mask(dim * dim, 0.0); // 3 *3

    for (size_t i = 0; i < dim; i++){
        size_t row, col;
        if (isUpper){
            row = i;
            col = (diagNum + i) % dim;
        }
        else{
            row = (diagNum + i) % dim;
            col = i;
        }
        mask[row * dim + col] = 1;
    }

    return mask;
}

Ciphertext<DCRTPoly> evalDotProduct(const Ciphertext<DCRTPoly>& q,
                                    const Ciphertext<DCRTPoly>& k,
                                    CryptoContext<DCRTPoly> cc,
                                    size_t tokenCount, size_t slots,
                                    size_t dim, lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys) {

    Ciphertext<DCRTPoly> result;
    size_t diagCount = min(tokenCount, dim);

    // Debug: Print input matrices
    {
        Plaintext ptQ, ptK;
        cc->Decrypt(keys.secretKey, q, &ptQ);
        cc->Decrypt(keys.secretKey, k, &ptK);
        ptQ->SetLength(9);
        ptK->SetLength(dim*dim);
        cout << "Input Q:\n" << ptQ->GetRealPackedValue() << endl;
        cout << "Input K:\n" << ptK->GetRealPackedValue() << endl;
    }

    for (size_t r = 0; r < diagCount; r++) {
        cout << "\n=== Processing diagonal " << r << " ===" << endl;
        Ciphertext<DCRTPoly> diagonalSum;

        for (size_t l = 0; l < tokenCount; l++) {
            cout << "-- Term l=" << l << " --" << endl;

            // Generate and debug U_{l-r}(Q) mask
            size_t u_diag = (dim + l - r) % dim;
            auto uMask = generateMask(true, u_diag, dim);
            cout << "U_" << u_diag << "(Q) mask: [";
            for (auto v : uMask) cout << v << " ";
            cout << "]" << endl;

            auto uQ = cc->EvalMult(q, (cc->MakeCKKSPackedPlaintext(uMask)));
            
            // Debug extracted diagonal
            Plaintext ptUQ;
            cc->Decrypt(keys.secretKey, uQ, &ptUQ);
            ptUQ->SetLength(dim*dim);
            cout << "Extracted U_" << u_diag << "(Q): " << ptUQ->GetRealPackedValue() << endl;

            // Rotate and debug
            auto rotatedUq = cc->EvalRotate(uQ, r);
            Plaintext ptRotUQ;
            cc->Decrypt(keys.secretKey, rotatedUq, &ptRotUQ);
            ptRotUQ->SetLength(dim*dim);
            cout << "After rotation by " << r << ": " << ptRotUQ->GetRealPackedValue() << endl;

            // Generate and debug L_l(K) mask
            auto lMask = generateMask(false, l, dim);
            cout << "L_" << l << "(K) mask: [";
            for (auto v : lMask) cout << v << " ";
            cout << "]" << endl;

            auto lK = cc->EvalMult(k, cc->MakeCKKSPackedPlaintext(lMask));
            
            // Debug extracted diagonal
            Plaintext ptLK;
            cc->Decrypt(keys.secretKey, lK, &ptLK);
            ptLK->SetLength(dim*dim);
            cout << "Extracted L_" << l << "(K): " << ptLK->GetRealPackedValue() << endl;

            // Multiply and debug
            auto product = cc->EvalMult(rotatedUq, lK);
            Plaintext ptProduct;
            cc->Decrypt(keys.secretKey, product, &ptProduct);
            ptProduct->SetLength(dim*dim);
            cout << "Product term: " << ptProduct->GetRealPackedValue() << endl;

            diagonalSum = (l == 0) ? product : cc->EvalAdd(diagonalSum, product);
            
            // Debug accumulated sum
            Plaintext ptSum;
            cc->Decrypt(keys.secretKey, diagonalSum, &ptSum);
            ptSum->SetLength(dim*dim);
            cout << "Current diagonal sum: " << ptSum->GetRealPackedValue() << endl;
        }

        // Debug before rotation
        Plaintext ptBeforeRot;
        cc->Decrypt(keys.secretKey, diagonalSum, &ptBeforeRot);
        ptBeforeRot->SetLength(dim*dim);
        cout << "\nDiagonal " << r << " before rotation: " << ptBeforeRot->GetRealPackedValue() << endl;

        if (r == 0) {
            result = diagonalSum;
        } else {
            result = (r == 0) ? diagonalSum : cc->EvalAdd(result, diagonalSum);

            
            // Debug after rotation
            Plaintext ptShifted;
            cc->Decrypt(keys.secretKey, result, &ptShifted);
            ptShifted->SetLength(dim*dim);
            cout << "After rotation by " << -r*dim << ": " << ptShifted->GetRealPackedValue() << endl;
            
            
        }

        // Debug current result
        Plaintext ptResult;
        cc->Decrypt(keys.secretKey, result, &ptResult);
        ptResult->SetLength(dim*dim);
        cout << "\nCurrent result matrix:\n" << ptResult->GetRealPackedValue() << endl;
    }

    // Final debug output
    Plaintext ptFinal;
    cc->Decrypt(keys.secretKey, result, &ptFinal);
    ptFinal->SetLength(dim*dim);
    cout << "\n=== Final Result ===\n" << ptFinal->GetRealPackedValue() << endl;

    return result;
}
/*
// Computes dot product between two ciphertexts
Ciphertext<DCRTPoly> evalDotProduct(const Ciphertext<DCRTPoly>& q,
                                    const Ciphertext<DCRTPoly>& k,
                                    CryptoContext<DCRTPoly> cc,
                                    size_t tokenCount, size_t slots,
                                    size_t dim) {
                         
    
    vector<Ciphertext<DCRTPoly>> score;

    for (size_t i = 0; i < slots; i += dim){ // for q[i]
        
        vector<double> mask(slots, 0.0);
        for (size_t m = i; m < i + 4; m++){
            mask[m] = 1;
        }                
        
        for (size_t j = 0; j < slots; j+=dim){// for k[i]
        
            // convert mask to plaintext
            
            // evalmult of q with mask
            //cc -> EvalMult(q, cc -> MakeCKKSPackedPlaintext(mask));
            // evalmult of q with evalrotate(j) of k
            //cc -> EvalRotate(k, j);

            auto scalar = cc -> EvalMult(cc -> EvalMult(q, cc -> MakeCKKSPackedPlaintext(mask)), cc -> EvalRotate(k, j - i));
            score.push_back(cc -> EvalSum(scalar, dim));
            
            
        }
           
        
    } 
    

    Ciphertext<DCRTPoly> packedScores = score[0];

    for (int l = 1; l < 9; l++){
        auto shifted = cc -> EvalRotate(score[l], -l);
        packedScores = cc -> EvalAdd(packedScores, shifted);
    }
    
    return packedScores;
}

*/

// Apply exp
Ciphertext<DCRTPoly> applyExp(const Ciphertext<DCRTPoly>& scores,
                                 size_t tokenCount, 
                                 CryptoContext<DCRTPoly> cc){
    // x^2 = scores * scores
    auto x2 = cc->EvalMultAndRelinearize(scores, scores);
    
    
    // x^3 = x^2 * scores
    auto x3 = cc->EvalMultAndRelinearize(x2, scores);
    

    // 0.5 * x^2
    auto x2Scaled = cc->EvalMult(x2, 0.5);
    //x2Scaled = cc->ModReduce(x2Scaled);

    // (1/6) * x^3 ≈ 0.1667
    auto x3Scaled = cc->EvalMult(x3, 1.0 / 6.0);
    //x3Scaled = cc->ModReduce(x3Scaled);

    // x + 0.5x^2
    auto partial = cc->EvalAdd(scores, x2Scaled);

    // + (1/6)x^3
    auto poly = cc->EvalAdd(partial, x3Scaled);
    // + 1
    return cc->EvalAdd(poly, 1.0);

}

Ciphertext<DCRTPoly> approximateInverse(const Ciphertext<DCRTPoly>& x, CryptoContext<DCRTPoly> cc, size_t iter = 1) {
    // Initial guess: use a simple negated linear approximation like -x
    Ciphertext<DCRTPoly> y = cc->EvalMult(x, -1.0);  // crude guess

    for (size_t i = 0; i < iter; i++) {
        auto xy = cc->EvalMultAndRelinearize(x, y);     
                       // xy
        auto two_minus_xy = cc->EvalSub(2.0, xy);        // 2 - xy
        y = cc->EvalMultAndRelinearize(y, two_minus_xy); 
                      // y * (2 - xy)
    }
    //y = cc->ModReduce(y);
    return y;  // Approximated 1/x
}

// Apply SoftMax
Ciphertext<DCRTPoly> applySoftMax(const Ciphertext<DCRTPoly>& scores,
                                   size_t tokenCount,
                                   CryptoContext<DCRTPoly> cc,
                                lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys) {
    int delta1 = 2;
    int delta2 = 1;

    // Shift: subtract an approximate max
    auto shift = cc->EvalSub(scores, 342.468); // crude shift approximation
    auto scaled = cc->EvalMult(shift, 1.0 / (delta1 * delta2));
    

    // --- Phase 1: Approximate exp(x / (d1 * d2)) ---
    auto exp1 = applyExp(scaled, tokenCount, cc);
    Plaintext exp1PT;
    cc->Decrypt(keys.secretKey, exp1, &exp1PT);
    exp1PT->SetLength(tokenCount);
    cout << "exp1 (after applyExp): " << exp1PT->GetRealPackedValue() << endl;


    // Raise to power delta1
    Ciphertext<DCRTPoly> expPower = exp1;
    for (int i = 1; i < delta1; i++) {
        expPower = cc->EvalMultAndRelinearize(expPower, exp1);
        
    }

    

    // Normalize by sum
    auto sumExp = cc->EvalSum(expPower, tokenCount);
    auto invSum = approximateInverse(sumExp, cc, 1);
    auto y = cc->EvalMult(expPower, invSum);
    

    y = cc -> EvalBootstrap(y);
    
    // --- Phase 2: Squaring and re-normalizing log2(delta2) times ---
    int steps = log2(delta2);
    

    for (int i = 0; i < steps; i++) {
        auto y2 = cc->EvalMult(y, y);
        
        auto sumY = cc->EvalSum(y2, tokenCount);
        auto invSumY = approximateInverse(sumY, cc, 2);

        // // to resolve drop error encountered
        // while (invSumY->GetLevel() > y2->GetLevel()) {
        //     invSumY = cc->ModReduce(invSumY);
        // }

        
        y = cc->EvalMult(y2, invSumY);
        
    }
    //y = cc->ModReduce(y);
    
    y = cc -> EvalBootstrap(y);

    return y;
}
/*
// Performs attention-weighted sum of values
void evalOutput(const Ciphertext<DCRTPoly> score,
                const Ciphertext<DCRTPoly>, 3>& v,
                vector<Ciphertext<DCRTPoly>>* output,
                CryptoContext<DCRTPoly> cc) {
    output->resize(3);
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            auto weighted = cc->EvalMult(score[i][j], v[j]);
            (*output)[i] = (j == 0) ? weighted : cc->EvalAdd((*output)[i], weighted);
        }
    }
}


// Adds residual connection to output
void evalOutputWithResidual(vector<Ciphertext<DCRTPoly>>* output,
                            const vector<Ciphertext<DCRTPoly>>& encPE,
                            CryptoContext<DCRTPoly> cc) {
    for (size_t i = 0; i < 3; i++) {
        (*output)[i] = cc->EvalAdd(encPE[i], (*output)[i]);
    }
}

// Applies feedforward network (x*W1)^2 * W2
void evalFeedForward(vector<Ciphertext<DCRTPoly>>* output,
                     const vector<double>& w1,
                     const vector<double>& w2,
                     CryptoContext<DCRTPoly> cc) {
    Plaintext ptxtW1 = cc->MakeCKKSPackedPlaintext(w1);
    Plaintext ptxtW2 = cc->MakeCKKSPackedPlaintext(w2);

    for (size_t i = 0; i < 3; i++) {
        (*output)[i] = cc->EvalMult((*output)[i], ptxtW1);
        (*output)[i] = cc->EvalMult((*output)[i], (*output)[i]);
        (*output)[i] = cc->EvalMult((*output)[i], ptxtW2);
    }
} 

*/
// Calculates the k-th diagonal of a matrix

int main() {
    EmbeddingMatrix embeddings = {
        {0.1, 0.3, 0.2, 0.05},  // "the"
        {0.4, 0.1, 0.2, 0.3},   // "cat"
        {0.3, 0.4, 0.1, 0.2}    // "sat"
    };

    std::vector<uint32_t> levelBudget = {10, 10};
    uint32_t levelsAfter = 20;

    // Use recommended bootstrap depth computation
    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
    uint32_t bootDepth = FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(secretKeyDist);  
    parameters.SetSecurityLevel(HEStd_NotSet);  

    parameters.SetFirstModSize(60);              
    parameters.SetScalingModSize(59);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    //parameters.SetMultiplicativeDepth(levelsAfter + bootDepth);
    parameters.SetMultiplicativeDepth(levelsAfter + bootDepth);
    parameters.SetRingDim(8192);                 

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    std::cout << "Starting bootstrap setup...\n";
    cc->EvalBootstrapSetup(levelBudget);
    std::cout << "Bootstrap setup completed.\n";



    size_t words = embeddings.size();
    size_t dim = embeddings[0].size();
    size_t total_slots = words * dim;

    EmbeddingMatrix peMatrix = addPositionalEncoding(embeddings);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalBootstrapKeyGen(keys.secretKey, cc->GetRingDimension() / 2);  
    
    vector<double> flattenPE = flattenMatrix(peMatrix);
    Plaintext ptxt;
    ptxt = MakeCKKSPackedTokens(flattenPE, cc);
    Ciphertext<DCRTPoly> encPE;
    encPE = cc->Encrypt(keys.publicKey, ptxt);

    cout << "Packed PE Plaintext: " << ptxt->GetRealPackedValue() << endl;


    EmbeddingMatrix W_Q = {{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2}, {1.3, 1.4, 1.5, 1.6}};
    EmbeddingMatrix W_K = {{0.2, 0.1, 0.4, 0.3}, {0.6, 0.5, 0.8, 0.7}, {1.0, 0.9, 1.2, 1.1}, {1.4, 1.3, 1.6, 1.5}};
    EmbeddingMatrix W_V = {{0.3, 0.4, 0.1, 0.2}, {0.7, 0.8, 0.5, 0.6}, {1.1, 1.2, 0.9, 1.0}, {1.5, 1.6, 1.3, 1.4}};

    
    
    vector<int32_t> rotIndices;
    for (size_t i = 0; i < total_slots; i++) rotIndices.push_back(i);
    cc->EvalAtIndexKeyGen(keys.secretKey, rotIndices);
    
    
    auto q = applyDiagonalProjection(encPE, W_Q, cc);
    auto k = applyDiagonalProjection(encPE, W_K, cc);
    auto v = applyDiagonalProjection(encPE, W_V, cc);


    Plaintext decScoreq;
    cc->Decrypt(keys.secretKey, q, &decScoreq);
    decScoreq->SetLength(embeddings.size() * embeddings.size()); // 9
    cout << "q slots: " << decScoreq->GetRealPackedValue() << endl;

    Plaintext decScorek;
    cc->Decrypt(keys.secretKey, k, &decScorek);
    decScorek->SetLength(embeddings.size() * embeddings.size()); // 9
    cout << "k slots: " << decScorek->GetRealPackedValue() << endl;
  

    vector<int32_t> negRotIndices;
    for (int l = 1; l < 12; l++) {
        negRotIndices.push_back(-l);
    }
    cc->EvalAtIndexKeyGen(keys.secretKey, negRotIndices);

    Ciphertext<DCRTPoly> score = evalDotProduct(q, k, cc, embeddings.size(), 12, 3, keys);

    Plaintext decScore;
    cc->Decrypt(keys.secretKey, score, &decScore);
    decScore->SetLength(embeddings.size() * embeddings.size()); // 9
    cout << "Score slots: " << decScore->GetRealPackedValue() << endl;

    //score = cc -> EvalBootstrap(score);
    Ciphertext<DCRTPoly> softMaxScore = applySoftMax(score, words, cc, keys);
    
    Plaintext decScore2;
    cc->Decrypt(keys.secretKey, softMaxScore, &decScore2);
    decScore2->SetLength(embeddings.size() * embeddings.size()); // 9
    cout << "Score slots: " << decScore2->GetRealPackedValue() << endl;


    
    /*
    vector<Ciphertext<DCRTPoly>> output;
    evalOutput(score, v, &output, cc);
    
    
    evalOutputWithResidual(&output, encPE, cc);

    vector<double> w1 = {0.3, 0.7, 0.2, 0.5};
    vector<double> w2 = {0.6, 0.4, 0.8, 0.1};
    evalFeedForward(&output, w1, w2, cc);

    */
    
}