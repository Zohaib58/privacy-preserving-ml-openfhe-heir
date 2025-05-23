#define PROFILE
#include "openfhe/pke/openfhe.h"
#include <cmath>
#include <array>
using namespace lbcrypto;
using namespace std;
using Vector = vector<double>; 
using EmbeddingMatrix = vector<Vector>;
using DotProdMatrix = vector<vector<Ciphertext<DCRTPoly>>>;


vector<double> calculateDiagonal(EmbeddingMatrix mat, int diagNum){
    int size = mat.size();
    vector<double> diagonalMat;
    for (int i = 0; i < size; i++)
    {
        diagonalMat.push_back(mat[i][(i + diagNum) % size]);
    }
    return diagonalMat;

}

EmbeddingMatrix addPositionalEncoding(EmbeddingMatrix embeddings){

    EmbeddingMatrix peMatrix;

    int words = embeddings.size();
    int dim = embeddings[0].size();

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

    for (int i = 0; i < words; i++){
        for (int j = 0; j < dim; j++)
        {
            peMatrix[i][j] += embeddings[i][j];
        }
    }

    return peMatrix;

}

array<Ciphertext<DCRTPoly>, 3> applyDiagonalProjection(vector<Ciphertext<DCRTPoly>> encPE, EmbeddingMatrix W_, CryptoContext<DCRTPoly> cc){

    size_t const words = 3;
    size_t const dim = 4;

    array<Ciphertext<DCRTPoly>, words> p;
        
    for (int i = 0; i < words; i++) {
        const auto& encTok = encPE[i];
        for (int j = 0; j < dim; j++){
            auto product = cc -> EvalMult((cc -> EvalRotate(encTok, j)), cc-> MakeCKKSPackedPlaintext(calculateDiagonal(W_, j)));
            p[i] = (j==0) ? product : cc -> EvalAdd(p[i], product) ;
        }
    }

    return p;

}

Ciphertext<DCRTPoly> evalDotProduct(Ciphertext<DCRTPoly> q, Ciphertext<DCRTPoly> k, CryptoContext<DCRTPoly> cc, size_t dim){
    return cc -> EvalSum(cc -> EvalMult(q, k), dim);
}

void evalOutput(vector<vector<Ciphertext<DCRTPoly>>> score, array<Ciphertext<DCRTPoly>, 3> v, vector<Ciphertext<DCRTPoly>>* output, CryptoContext<DCRTPoly> cc){
    
    output->resize(3);
    for (size_t i = 0; i < 3; i++){
        for (size_t j = 0; j < 3; j++){
            auto weighted = cc -> EvalMult(score[i][j], v[j]); 
            (*output)[i] = (j == 0) ? weighted : cc -> EvalAdd((*output)[i], weighted);
        }
    }
}

void evalOutputWithResidual(vector<Ciphertext<DCRTPoly>>* output, vector<Ciphertext<DCRTPoly>> encPE, CryptoContext<DCRTPoly> cc){
    for (size_t i = 0; i < 3; i++){
       (*output)[i] = cc -> EvalAdd(encPE[i], (*output)[i]);
    }
}

int main(){
    
    // Step 1: Tokenized sentences
    EmbeddingMatrix embeddings = {
        {0.1, 0.3, 0.2, 0.05},  // "the"
        {0.4, 0.1, 0.2, 0.3},   // "cat"
        {0.3, 0.4, 0.1, 0.2}    // "sat"
    };  

    uint32_t scaleModSize = 50;
    uint32_t multDepth = 6;
    u_int32_t batchSize = 4;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetBatchSize(batchSize);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc -> Enable(PKE);
    cc -> Enable(KEYSWITCH);
    cc -> Enable(LEVELEDSHE);
    cc -> Enable(ADVANCEDSHE);
   
    size_t words = embeddings.size();
    size_t dim = embeddings[0].size();

    // postional encoding]
    EmbeddingMatrix peMatrix;
    if (!embeddings.empty()){
        peMatrix = addPositionalEncoding(embeddings);
    }

    if (!peMatrix.empty()){
        auto keys = cc -> KeyGen();
        cc -> EvalMultKeyGen(keys.secretKey);
            
        vector<Ciphertext<DCRTPoly>> encPE;
        for (int i = 0; i < words; i++){
        
            Plaintext ptxt = cc -> MakeCKKSPackedPlaintext(peMatrix[i]);    
            auto enc = cc -> Encrypt(keys.publicKey, ptxt);
            encPE.push_back(enc);
        }


        // start working on projects via diagonal matrices
        EmbeddingMatrix W_Q =
        {
            {0.1, 0.2, 0.3, 0.4},
            {0.5, 0.6, 0.7, 0.8},
            {0.9, 1.0, 1.1, 1.2},
            {1.3, 1.4, 1.5, 1.6}
        };

        EmbeddingMatrix W_K =
        {
            {0.2, 0.1, 0.4, 0.3},
            {0.6, 0.5, 0.8, 0.7},
            {1.0, 0.9, 1.2, 1.1},
            {1.4, 1.3, 1.6, 1.5}

        };
  
        EmbeddingMatrix W_V =
        {
            {0.3, 0.4, 0.1, 0.2},
            {0.7, 0.8, 0.5, 0.6},
            {1.1, 1.2, 0.9, 1.0},
            {1.5, 1.6, 1.3, 1.4}
        };

        
        vector<int32_t> rotIndicesQ;
        for (int i = 0; i < dim; i++){
            rotIndicesQ.push_back(i);
        }
        cc -> EvalAtIndexKeyGen(keys.secretKey, rotIndicesQ);

        array<Ciphertext<DCRTPoly>, 3> q = applyDiagonalProjection(encPE, W_Q, cc);
        array<Ciphertext<DCRTPoly>, 3> k = applyDiagonalProjection(encPE, W_K, cc);
        array<Ciphertext<DCRTPoly>, 3> v = applyDiagonalProjection(encPE, W_V, cc);
        
        vector<vector<Ciphertext<DCRTPoly>>> score(words, vector<Ciphertext<DCRTPoly>>(words));
        for (size_t i = 0; i < words; i++){
            for (size_t j = 0; j < words; j++){
                score[i][j] = evalDotProduct(q[i], k[j], cc, words);
            }  
        }

        vector<Ciphertext<DCRTPoly>> output;
        evalOutput(score, v, &output, cc);
        evalOutputWithResidual(&output, encPE, cc);

        vector<int32_t> rotIndices;
        for (size_t i = 1; i < dim; i *= 2){
            rotIndices.push_back(i);
        }
        cc -> EvalAtIndexKeyGen(keys.secretKey, rotIndices);
        
        if (!output.empty()){

            for (int i = 0; i < words; i++){
                Plaintext decrypted;
                cc -> Decrypt(keys.secretKey, output[i], &decrypted);
                decrypted -> SetLength(dim);
                cout << decrypted->GetRealPackedValue() << endl;


            }
            

        }

    }

      
}
    


        

        