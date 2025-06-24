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

EmbeddingMatrix transposeMatrix(EmbeddingMatrix matrix){
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    
    EmbeddingMatrix transposeMatrix(cols, Vector(rows));

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            transposeMatrix[j][i] = matrix[i][j]; 
        }
    }

    return transposeMatrix;
}

vector<double> flattenMatrixUpperDiag(const vector<vector<double>>& A) {
    int rows = A.size();
    if (rows == 0) return {};
    int cols = A[0].size();
    int diags = min(rows, cols);  // Number of upper diagonals
    int maxDim = max(rows, cols); // Length of each diagonal
    
    vector<double> flattenedDiagonals;
    
    for (int k = 0; k < diags; k++) {
        vector<double> diag(maxDim);
        for (int t = 0; t < maxDim; t++) {
            diag[t] = A[t % rows][(k + t) % cols];
        }
        flattenedDiagonals.insert(flattenedDiagonals.end(), diag.begin(), diag.end());
    }
    
    return flattenedDiagonals;
}

vector<double> flattenMatrixLowerDiag(const vector<vector<double>>& A) {
    int rows = A.size();
    if (rows == 0) return {};
    int cols = A[0].size();
    int diags = min(rows, cols);  // Number of lower diagonals
    int maxDim = max(rows, cols); // Length of each diagonal
    
    vector<double> flattenedDiagonals;
    
    for (int k = 0; k < diags; k++) {
        vector<double> diag(maxDim);
        for (int t = 0; t < maxDim; t++) {
            diag[t] = A[(k + t) % rows][t % cols];
        }
        flattenedDiagonals.insert(flattenedDiagonals.end(), diag.begin(), diag.end());
    }
    
    return flattenedDiagonals;
}

vector<double> flattenMatrixInterlacedDiagEncoding(vector<double> flatDE, size_t tokens, size_t dim, size_t diagsCount){
    
    vector<double> flatIDE;

    for (size_t i = 0; i < tokens; i++){
        for (size_t j = 0; j < diagsCount; j++){
            flatIDE.push_back(flatDE[j * tokens + i]);
        }
    }
    return flatIDE;
}

Ciphertext<DCRTPoly> product(Ciphertext<DCRTPoly> ctxt, Plaintext ptxt, size_t diags, size_t dim, CryptoContext<DCRTPoly> cc, 
                            lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys, size_t batchSize) {

    Ciphertext<DCRTPoly> all_diagonals;
    bool first_diagonal = true;


    for (size_t diagOffset = 0; diagOffset < diags; diagOffset++) {
        
        // Rotate X to align L₀, L₁, ..., L_{c-1} to match U₀...U_{c-1}
        Ciphertext<DCRTPoly> rotatedX = cc->EvalRotate(ctxt, diagOffset * dim);

        // Elementwise product
        auto mult = cc->EvalMult(ptxt, rotatedX);

        // Aggregate within this diagonal group
        auto sum = mult;

        vector<double> mask(batchSize, 0.0);
        for (size_t m = 0; m < dim; m++)  mask[m] = 1;

        for (size_t i = 1; i < dim; i++) {
            auto rot = cc->EvalRotate(mult, i * dim);
            sum = cc->EvalAdd(sum, rot);
           
        }

        Plaintext sumTemp;
            cc->Decrypt(keys.secretKey, sum, &sumTemp);
            sumTemp->SetLength(batchSize);  // or actual length of packed data
            //cout << "Diagonal: " << sumTemp->GetRealPackedValue() << endl;

        sum = cc -> EvalMult(sum, cc -> MakeCKKSPackedPlaintext(mask));
        
        sum = cc -> EvalRotate(sum, int(-(diagOffset * dim)));

        if (first_diagonal) {
            all_diagonals = sum;  
            first_diagonal = false;
        } else {
            all_diagonals = cc->EvalAdd(all_diagonals, sum);
        }

        
        Plaintext temp;
        cc->Decrypt(keys.secretKey, all_diagonals, &temp);
        temp->SetLength(batchSize);  // or actual length of packed data
        //cout << "Diagonal: " << temp->GetRealPackedValue() << endl;

    }

    return all_diagonals;
}
Plaintext MakeCKKSPackedTokens(vector<double> flattenPE, CryptoContext<DCRTPoly> cc){
    return cc->MakeCKKSPackedPlaintext(flattenPE);
}


int main() {
    EmbeddingMatrix embeddings = {
        {0.1, 0.3, 0.2, 0.05},  // "the"
        {0.4, 0.1, 0.2, 0.3},   // "cat"
        {0.3, 0.4, 0.1, 0.2},    // "sat"
        {0.2, 0.5, 0.3, 0.4}    // "runs"
    };

    // Parameters aligned with your softmax production setup
    //std::vector<uint32_t> levelBudget = {10, 10};
    //uint32_t levelsAfter = 20;
    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;

    //uint32_t bootDepth = FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetSecurityLevel(HEStd_NotSet);        // You can use NotSet if you want manual tuning
    parameters.SetFirstModSize(60);
    parameters.SetScalingModSize(35);                      // Updated from 59 → 35 as per your prod config
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(25); // 30
    parameters.SetRingDim(8192);
    size_t batchSize = 16;                         // Your production ring dimension
    parameters.SetBatchSize(batchSize);                         // SIMD capacity as per softmax setup
                 

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    size_t wordsE = embeddings.size();
    size_t dimE = embeddings[0].size();
    size_t total_slots = wordsE * dimE;
    size_t diagsE = min(wordsE, dimE);

    EmbeddingMatrix peEmbeddings = addPositionalEncoding(embeddings);
    cout << "peEmbeddings: " << peEmbeddings << endl;


    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    //cc->EvalBootstrapKeyGen(keys.secretKey, cc->GetRingDimension() / 2);  
    
    EmbeddingMatrix EmbeddingsT = transposeMatrix(peEmbeddings);
    size_t wordsET = EmbeddingsT.size();
    size_t dimET = EmbeddingsT[0].size();
    size_t diagsET = min(wordsET, dimET);

    
    vector<double> flattenDiagEmbeddings = flattenMatrixLowerDiag(peEmbeddings);
    vector<double> flattenDiagEmbeddingsT = flattenMatrixLowerDiag(EmbeddingsT);
    //cout << "flatteIDiagEmbeddingsT" << flattenDiagEmbeddingsT << endl;

    /*
    vector<double> flattenIDiagEmbeddings = flattenMatrixInterlacedDiagEncoding(flattenDiagEmbeddings, wordsE, dimE, diagsE);
    
    */
    
    /*
    vector<double> flattenIDiagEmbeddingsT = flattenMatrixInterlacedDiagEncoding(flattenDiagEmbeddingsT, wordsET, dimET, diagsET);

    */
    

    Plaintext px;
    px = MakeCKKSPackedTokens(flattenDiagEmbeddings, cc);
    
    Plaintext pxT;
    pxT = MakeCKKSPackedTokens(flattenDiagEmbeddingsT, cc);
    
    Ciphertext<DCRTPoly> cx;
    cx = cc->Encrypt(keys.publicKey, px);

    Ciphertext<DCRTPoly> cxT;
    cxT = cc->Encrypt(keys.publicKey, pxT);

    Plaintext deccxT;
    cc->Decrypt(keys.secretKey, cxT, &deccxT);
    deccxT->SetLength(embeddings.size() * embeddings[0].size()); // 9
    cout << "cxT: " << deccxT->GetRealPackedValue() << endl;

    EmbeddingMatrix W_Q = {{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2}, {1.3, 1.4, 1.5, 1.6}};
    EmbeddingMatrix W_K = {{0.2, 0.1, 0.4, 0.3}, {0.6, 0.5, 0.8, 0.7}, {1.0, 0.9, 1.2, 1.1}, {1.4, 1.3, 1.6, 1.5}};
    EmbeddingMatrix W_V = {{0.3, 0.4, 0.1, 0.2}, {0.7, 0.8, 0.5, 0.6}, {1.1, 1.2, 0.9, 1.0}, {1.5, 1.6, 1.3, 1.4}};

    EmbeddingMatrix QT = transposeMatrix(W_Q);
    size_t rowQT = QT.size();
    size_t dimQT = QT[0].size();
    
    //cout << "Q Ttranposed " << QT << endl;
    vector<double> flattenDiagQT = flattenMatrixUpperDiag(QT);
    //cout << "flatten Diag QT " << flattenDiagQT << endl;
    
    
    size_t diagsQT = min(W_Q.size(), W_Q[0].size());
    vector<double> flattenIDiagQT = flattenMatrixInterlacedDiagEncoding(flattenDiagQT, rowQT, dimQT, diagsQT);
    cout << "flattenIDiagQT" << flattenIDiagQT << endl;

    
    

    Plaintext qtptxt;
    qtptxt = cc -> MakeCKKSPackedPlaintext(flattenIDiagQT);

    vector<double> decodedVector = qtptxt->GetRealPackedValue();
    //cout << "qtptxt " << decodedVector << endl;

    vector<int32_t> rotIndices;
    for (size_t i = 0; i < total_slots; i++) rotIndices.push_back(i);
    cc->EvalAtIndexKeyGen(keys.secretKey, rotIndices);

    size_t dimP = dimET;
    size_t diagsP = rowQT;

    
    vector<int32_t> negRotIndices;
    for (int l = 1; l < 16; l++) {
        negRotIndices.push_back(-l);
    }
    cc->EvalAtIndexKeyGen(keys.secretKey, negRotIndices);

    Ciphertext<DCRTPoly> cqt = product(cxT, qtptxt, diagsP, dimP, cc, keys, batchSize);

    Plaintext temp;
    cc->Decrypt(keys.secretKey, cqt, &temp);
    temp->SetLength(batchSize);  // or the actual expected length
    cout << "cqt" << temp->GetRealPackedValue() << endl;

    size_t rowK = W_K.size();
    size_t dimK = W_K[0].size();
    
    vector<double> flattenDiagK = flattenMatrixUpperDiag(W_K);
    size_t diagsK = min(W_K.size(), W_K[0].size());
    vector<double> flattenIDiagK = flattenMatrixInterlacedDiagEncoding(flattenDiagK, rowK, dimK, diagsK);
    Plaintext kptxt = cc -> MakeCKKSPackedPlaintext(flattenIDiagK);
    cout << "flattenIDiagK" << flattenIDiagK << endl;


    size_t dimPK = dimE;
    size_t diagsPK = rowK;
    Ciphertext<DCRTPoly> ck = product(cxT, kptxt, diagsPK, dimPK, cc, keys, batchSize);
    Plaintext tempk;
    cc->Decrypt(keys.secretKey, ck, &tempk);
    temp->SetLength(batchSize);  // or the actual expected length
    cout << "ck" << tempk->GetRealPackedValue() << endl;
    
    
   


    //Ciphertext<DCRTPoly> score = evalDotProduct(q, k, cc, embeddings.size(), 12, dim);

    /*
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

    
    */
    

    
    /*
    vector<Ciphertext<DCRTPoly>> output;
    evalOutput(score, v, &output, cc);
    
    
    evalOutputWithResidual(&output, encPE, cc);

    vector<double> w1 = {0.3, 0.7, 0.2, 0.5};
    vector<double> w2 = {0.6, 0.4, 0.8, 0.1};
    evalFeedForward(&output, w1, w2, cc);

    */
    
}


/*
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



// Computes dot product between two ciphertexts
Ciphertext<DCRTPoly> evalDotProduct(const Ciphertext<DCRTPoly>& q,
                                    const Ciphertext<DCRTPoly>& k,
                                    CryptoContext<DCRTPoly> cc,
                                    size_t tokenCount, size_t slots,
                                    size_t dim) {
                         
    
    vector<Ciphertext<DCRTPoly>> dotProducts;

    for (size_t i = 0; i < tokenCount; ++i) {
        // Create a mask to extract qᵢ
        vector<double> qMask(slots, 0.0);
        for (size_t d = 0; d < dim; ++d) {
            qMask[i * dim + d] = 1.0;
        }

        auto q_i = cc->EvalMult(q, cc->MakeCKKSPackedPlaintext(qMask));

        for (size_t j = 0; j < tokenCount; ++j) {
            // Create a mask to extract kⱼ
            vector<double> kMask(slots, 0.0);
            for (size_t d = 0; d < dim; ++d) {
                kMask[j * dim + d] = 1.0;
            }

            auto k_j = cc->EvalMult(k, cc->MakeCKKSPackedPlaintext(kMask));

            // Multiply qᵢ and kⱼ element-wise
            auto prod = cc->EvalMult(q_i, k_j);

            // Sum over dimensions to compute dot(qᵢ, kⱼ)
            auto dot = cc->EvalSum(prod, dim);
            dotProducts.push_back(dot);
        }
    }

    // Pack all dot products into one ciphertext (row-major layout)
    Ciphertext<DCRTPoly> packed = dotProducts[0];
    for (size_t l = 1; l < dotProducts.size(); ++l) {
        auto shifted = cc->EvalRotate(dotProducts[l], -static_cast<int>(l));
        packed = cc->EvalAdd(packed, shifted);
    }

    return packed;

}


*/


/*
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
    

    //y = cc -> EvalBootstrap(y);
    
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
    
    //y = cc -> EvalBootstrap(y);

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

