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


vector<double> calculateDiagonal(const EmbeddingMatrix& W, int diagNum) {
    size_t words = W.size();          // number of rows
    size_t dim = W[0].size();         // number of columns
    size_t total_slots = words * dim;

    vector<double> diag(total_slots, 0.0);

    for (size_t row = 0; row < words; ++row) {
        for (size_t col = 0; col < dim; ++col) {
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
    const size_t rows = words * dim;
    const size_t total_slots = words * dim;

    vector<Plaintext> diagMat;
    for (size_t h = 0; h < total_slots; h++) {
        diagMat.push_back(cc->MakeCKKSPackedPlaintext(calculateDiagonal(W_, h)));
    }
    
    Ciphertext<DCRTPoly> p;
    for (int j = 0; j < total_slots; j++){
        auto product = cc -> EvalMult(cc -> EvalRotate(encPE, j), diagMat[j]); 
        p = (j==0) ? product : cc -> EvalAdd(p, product);
    }
 
    return p;
}



// Computes dot product between two ciphertexts
Ciphertext<DCRTPoly> evalDotProduct(const Ciphertext<DCRTPoly>& q,
                                    const Ciphertext<DCRTPoly>& k,
                                    CryptoContext<DCRTPoly> cc,
                                    size_t dim) {
                         
    vector<Ciphertext<DCRTPoly>> score(9);

    int count = 0;
    for (size_t i = 0; i < 12; i += 4){ // for q[i]
        
        vector<double> mask(12, 0.0);
        for (size_t k = i; k < i + 4; k++){
                mask[i] = 1;
        }
        
        
        // vec to store dot products
        int count2 = 0;
        for (size_t j = 0; j < 12; j+=4){// for k[i]
            
            

            // convert mask to plaintext
            
            // evalmult of q with mask
            //cc -> EvalMult(q, cc -> MakeCKKSPackedPlaintext(mask));
            // evalmult of q with evalrotate(j) of k
            //cc -> EvalRotate(k, j);

            auto scalar = cc -> EvalMult(cc -> EvalMult(q, cc -> MakeCKKSPackedPlaintext(mask)), cc -> EvalRotate(k, j));
            score[count + count2] = scalar;
            ++count2;
            
        }
        count += count2 + 1;
        // convert to plainText nd store as slots in the vector   
        
    }
    

    Ciphertext<DCRTPoly> score2 = score[0];

    for (int l = 1; l < 9; l++){
        auto shifted = cc -> EvalRotate(score[l], -l);
        score2 = cc -> EvalAdd(score2, shifted);
    }
    
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

    uint32_t scaleModSize = 50;
    uint32_t multDepth = 6;
    uint32_t batchSize = 16; // to accomodate all tokens into single ciphertext

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetBatchSize(batchSize);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    size_t words = embeddings.size();
    size_t dim = embeddings[0].size();

    EmbeddingMatrix peMatrix = addPositionalEncoding(embeddings);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    vector<int32_t> rotIndices;
    for (size_t i = 0; i < dim; i++) rotIndices.push_back(i);
    cc->EvalAtIndexKeyGen(keys.secretKey, rotIndices);

    
    vector<double> flattenPE = flattenMatrix(peMatrix);
    Plaintext ptxt;
    ptxt = MakeCKKSPackedTokens(flattenPE, cc);
    Ciphertext<DCRTPoly> encPE;
    encPE = cc->Encrypt(keys.publicKey, ptxt);

    cout << "Packed PE Plaintext: " << ptxt->GetRealPackedValue() << endl;


    EmbeddingMatrix W_Q = {{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2}, {1.3, 1.4, 1.5, 1.6}};
    EmbeddingMatrix W_K = {{0.2, 0.1, 0.4, 0.3}, {0.6, 0.5, 0.8, 0.7}, {1.0, 0.9, 1.2, 1.1}, {1.4, 1.3, 1.6, 1.5}};
    EmbeddingMatrix W_V = {{0.3, 0.4, 0.1, 0.2}, {0.7, 0.8, 0.5, 0.6}, {1.1, 1.2, 0.9, 1.0}, {1.5, 1.6, 1.3, 1.4}};

    
    auto q = applyDiagonalProjection(encPE, W_Q, cc);
    auto k = applyDiagonalProjection(encPE, W_K, cc);
    auto v = applyDiagonalProjection(encPE, W_V, cc);

    
    Ciphertext<DCRTPoly> score = evalDotProduct(q, k, cc, dim * 3);

    
    /*
    vector<Ciphertext<DCRTPoly>> output;
    evalOutput(score, v, &output, cc);
    
    
    evalOutputWithResidual(&output, encPE, cc);

    vector<double> w1 = {0.3, 0.7, 0.2, 0.5};
    vector<double> w2 = {0.6, 0.4, 0.8, 0.1};
    evalFeedForward(&output, w1, w2, cc);

    */
    
}