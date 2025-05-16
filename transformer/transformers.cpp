#define PROFILE
#include "openfhe/pke/openfhe.h"
#include <cmath>
using namespace lbcrypto;
using namespace std;
using Vector = vector<double>; 
using EmbeddingMatrix = vector<Vector>;
using DotProdMatrix = vector<vector<Ciphertext<DCRTPoly>>>;

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
   
    int8_t words = 0;
    int8_t dim = 0;

    // postional encoding
    if (!embeddings.empty()){
        words = embeddings.size();
        dim = embeddings[0].size();
    }

    EmbeddingMatrix peMatrix;

    if (words && dim) {
        
        for (int i = 0; i < words; i++)
        {
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

    }

    if (peMatrix.size() > 0) {
        for (int i = 0; i < words; i++){
            for (int j = 0; j < dim; j++)
            {
                peMatrix[i][j] += embeddings[i][j];
            }
        }

        auto keys = cc -> KeyGen();
        cc -> EvalMultKeyGen(keys.secretKey);
            
        vector<Ciphertext<DCRTPoly>> encPE;
        for (int i = 0; i < words; i++){
        
            Plaintext ptxt = cc -> MakeCKKSPackedPlaintext(peMatrix[i]);    
            auto enc = cc -> Encrypt(keys.publicKey, ptxt);
            encPE.push_back(enc);
        }

        vector<int32_t> rotIndices;
        for (int i = 1; i < dim; i *= 2){
            rotIndices.push_back(i);
        }
        cc -> EvalAtIndexKeyGen(keys.secretKey, rotIndices);

        DotProdMatrix dotProdMatrix(words, vector<Ciphertext<DCRTPoly>>(words));
        if (peMatrix.size() > 0){
            for (int i = 0; i < words; i++){
                for (int j =  0; j < words; j++){
                    dotProdMatrix[i][j] = cc -> EvalSum((cc -> EvalMult(encPE[i], encPE[j])), dim);
                }
            }
        }
        
        vector<Ciphertext<DCRTPoly>> updEncVec(words);
        if (!dotProdMatrix.empty() && !dotProdMatrix[0].empty()){
            for (int i = 0; i < words; i++){
                for (int j =  0; j < words; j++){
                    auto product = cc -> EvalMult(dotProdMatrix[i][j], encPE[j]);

                    if (j == 0){
                        updEncVec[i] = product;
                    }
                    else{
                        updEncVec[i] = cc -> EvalAdd(updEncVec[i], product);
                    }
                    
                }
            }
        }

        if (!updEncVec.empty()){

            for (int i = 0; i < words; i++){
                Plaintext decrypted;
                cc -> Decrypt(keys.secretKey, updEncVec[i], &decrypted);
                decrypted -> SetLength(dim);
                cout << decrypted->GetRealPackedValue() << endl;


            }
            

        }

    }

      
}