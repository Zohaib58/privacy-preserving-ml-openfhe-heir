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



// Masks

Plaintext GenMask(CryptoContext<DCRTPoly> cc, size_t slotCount, const vector<size_t>& activeSlots) {
    vector<double> mask(slotCount, 0.0);
    for (auto i : activeSlots) {
        if (i < slotCount)
            mask[i] = 1.0;
    }
    return cc->MakeCKKSPackedPlaintext(mask);
}

std::vector<size_t> RotateIndices(const std::vector<size_t>& indices, size_t rot, size_t slotCount) {
    std::vector<size_t> rotated;
    for (size_t i : indices) {
        rotated.push_back((i + rot) % slotCount);
    }
    return rotated;
}


vector<vector<Plaintext>> generateAllMasks(CryptoContext<DCRTPoly> cc, size_t n, size_t c){
    size_t totalSlots = n * c;

    vector<vector<Plaintext>> masks(n);

    for (size_t l = 1; l < n; ++l) {
       
        size_t l_mod_c = l % c;  // [l]_c

        // μ_ℓ,0: First (n-ℓ) entries for [ℓ]_c ≤ r < c
        vector<size_t> mu0Indices;
        for (size_t r = l_mod_c; r < c; ++r) {
            for (size_t t = 0; t < n - l; ++t) {
                mu0Indices.push_back(r * n + t);
            }
        }

        // μ_ℓ,1: First (n-ℓ) entries for 0 ≤ r < [ℓ]_c
        vector<size_t> mu1Indices;
        for (size_t r = 0; r < l_mod_c; ++r) {
            for (size_t t = 0; t < n - l; ++t) {
                mu1Indices.push_back(r * n + t);
            }
        }

        // μ_ℓ,2: Last ℓ entries for all r
        vector<size_t> mu2Indices;
        for (size_t r = 0; r < c; ++r) {
            for (size_t t = n - l; t < n; ++t) {
                mu2Indices.push_back(r * n + t);
            }
        }

        cout << "mu2 Ind" << mu2Indices << endl;

        auto mu2Rotated = RotateIndices(mu2Indices, -n, totalSlots);

        cout << "mu2 Rot" << mu2Rotated << endl;


        masks[l].push_back(GenMask(cc, totalSlots, mu0Indices));  // μℓ,0
        masks[l].push_back(GenMask(cc, totalSlots, mu1Indices));  // μℓ,1
        masks[l].push_back(GenMask(cc, totalSlots, mu2Rotated));  // μℓ,2
    }

    return masks;
    
}

vector<Ciphertext<DCRTPoly>> productct(vector<Ciphertext<DCRTPoly>> ctxt1, vector<Ciphertext<DCRTPoly>> ctxt2, CryptoContext<DCRTPoly> cc, 
                            lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys, size_t batchSize, size_t m, size_t n, size_t c) {
    
    Plaintext tempCtxt;
    cc->Decrypt(keys.secretKey, ctxt1[0], &tempCtxt);
    tempCtxt->SetLength(batchSize);  // or the actual expected length
    cout << "ctxt 1" << tempCtxt->GetRealPackedValue() << endl;

    
    
    Plaintext tempCtxt1;
    cc->Decrypt(keys.secretKey, ctxt2[1], &tempCtxt1);
    tempCtxt1->SetLength(batchSize);  // or the actual expected length
    cout << "ctxt 2 1" << tempCtxt1->GetRealPackedValue() << endl;

    Plaintext tempCtxt2;
    cc->Decrypt(keys.secretKey, ctxt2[2], &tempCtxt2);
    tempCtxt2->SetLength(batchSize);  // or the actual expected length
    cout << "ctxt 2 2" << tempCtxt2->GetRealPackedValue() << endl;

    
    
    size_t m_ = m/c;

    vector<Ciphertext<DCRTPoly>> ctCResults(m_);
    
    vector<vector<Ciphertext<DCRTPoly>>> Ctjl(n);

    for (size_t j = 0; j < m_; j++){
        
        ctCResults[j] = cc -> EvalMult(ctxt1[j], ctxt2[0]);
        Ctjl[0].push_back(ctCResults[j]);
        auto ctxt = Ctjl[0];  

        for (size_t l = 1; l < n; l++){
            int lc = l % c;
            int rot = -static_cast<int>(n * lc) + (l);

            /*
            Plaintext tempCtxt;
            cc->Decrypt(keys.secretKey, ctxt1[j], &tempCtxt);
            tempCtxt->SetLength(batchSize);  // or the actual expected length
            cout << "ctxt " << j << tempCtxt->GetRealPackedValue() << endl;
            */
            
            auto rotated = cc -> EvalRotate(ctxt1[j], rot);

            /*
            Plaintext tempRotprd;
            cc->Decrypt(keys.secretKey, rotated, &tempRotprd);
            tempRotprd->SetLength(batchSize);  // or the actual expected length
            cout << "ctxt rot " << j << tempRotprd->GetRealPackedValue() << endl;
            
            */
            
            auto product = cc->EvalMult(rotated, ctxt2[l]);
            Ctjl[l].push_back(product); // we have green row/ product of 0th diag 

            /*
            Plaintext tempRot;
            cc->Decrypt(keys.secretKey, product, &tempRot);
            tempRot->SetLength(batchSize);  // or the actual expected length
            cout << "product " << tempRot->GetRealPackedValue() << endl;
            
            */
            
            
            
        }
    }

    /*
    std::vector<std::vector<std::vector<double>>> decryptedResults;

    
    // Iterate over each row (first dimension)
    for (const auto& row : Ctjl) {
        std::vector<std::vector<double>> decryptedRow;

        // Iterate over each ciphertext in the row (second dimension)
        for (const auto& ct : row) {
            // Decrypt the ciphertext
            Plaintext pt;
            cc->Decrypt(keys.secretKey, ct, &pt);

            // Extract the plaintext values (assuming packed encoding)
            std::vector<double> values = pt->GetRealPackedValue();
            decryptedRow.push_back(values);
        }

        decryptedResults.push_back(decryptedRow);
    }

    // Print decrypted values (for debugging)
    for (size_t i = 0; i < decryptedResults.size(); ++i) {
        for (size_t j = 0; j < decryptedResults[i].size(); ++j) {
            std::cout << "Ctjl[" << i << "][" << j << "]: ";
            for (double val : decryptedResults[i][j]) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    
    */
    
    vector<vector<array<Ciphertext<DCRTPoly>, 4>>> maskedParts(m_);
    auto masks = generateAllMasks(cc, n, c);  

    /*
    for (size_t l = 1; l < masks.size(); l++) {
        for (size_t j = 0; j < masks[1].size(); j++){
            cout << l << ":" << j << masks[l][j]->GetRealPackedValue() << endl;
        }
    }


    
    */
    
    for (size_t l = 1; l < n; l++) {
        for (size_t j = 0; j < m_; j++) {
            auto& ctxt = Ctjl[l][j];
            /*
            Plaintext tempCtxt;
            cc->Decrypt(keys.secretKey, ctxt, &tempCtxt);
            tempCtxt->SetLength(batchSize);  // or the actual expected length
            cout << "ctxt" << tempCtxt->GetRealPackedValue() << endl;
            */
            auto masked0 = cc->EvalMult(ctxt, masks[l][0]);
            /*
            Plaintext tempm0;
            cc->Decrypt(keys.secretKey, masked0, &tempm0);
            tempm0->SetLength(batchSize);  // or the actual expected length
            cout << "tempm0" << tempm0->GetRealPackedValue() << endl;

            */
            auto masked1 = cc->EvalMult(ctxt, masks[l][1]);
            /*
            Plaintext tempm1;
            cc->Decrypt(keys.secretKey, masked1, &tempm1);
            tempm1->SetLength(batchSize);  // or the actual expected length
            cout << "tempm1" << tempm1->GetRealPackedValue() << endl;
            */
            auto masked2 = cc->EvalMult(ctxt, masks[l][2]);
            
            /*
            Plaintext tempm1;
            cc->Decrypt(keys.secretKey, masks[l][2], &tempm1);
            tempm1->SetLength(batchSize);  // or the actual expected length
            cout << "tempm1" << tempm1->GetRealPackedValue() << endl;
            
            */
            

            auto residual = cc->EvalSub(ctxt, masked0);
            residual = cc->EvalSub(residual, masked1);
            residual = cc->EvalSub(residual, masked2);
            
            maskedParts[j].push_back({masked0, masked1, masked2, residual});


        }
    }

    /*
    
                // Loop through each j (outer loop)
            for (size_t j = 0; j < maskedParts.size(); ++j) {
                std::cout << "--- j = " << j << " ---\n";
                
                // Loop through each ℓ (inner loop)
                for (size_t l = 0; l < maskedParts[j].size(); ++l) {
                    std::cout << "  ℓ = " << l << ":\n";
                    
                    // Get the 4 components for this (j,ℓ) pair
                    const auto& [masked0, masked1, masked2, residual] = maskedParts[j][l];
                    
                    // Decrypt each component (assuming you have access to 'cc' and 'keys')
                    Plaintext pt0, pt1, pt2, pt_res;
                    cc->Decrypt(keys.secretKey, masked0, &pt0);
                    cc->Decrypt(keys.secretKey, masked1, &pt1);
                    cc->Decrypt(keys.secretKey, masked2, &pt2);
                    cc->Decrypt(keys.secretKey, residual, &pt_res);
                    
                    // Set length for printing (adjust if needed)
                    pt0->SetLength(batchSize);
                    pt1->SetLength(batchSize);
                    pt2->SetLength(batchSize);
                    pt_res->SetLength(batchSize);
                    
                    // Print all 4 values
                    std::cout << "    masked0:   " << pt0->GetRealPackedValue() << "\n";
                    std::cout << "    masked1:   " << pt1->GetRealPackedValue() << "\n";
                    std::cout << "    masked2:   " << pt2->GetRealPackedValue() << "\n";
                    std::cout << "    residual:  " << pt_res->GetRealPackedValue() << "\n\n";
                }
            }


    
    
    */
    
    for (size_t j = 0; j < m_; j++) {
        Ciphertext<DCRTPoly> ctCjPrime;
        Ciphertext<DCRTPoly> ctCjPrimePrime;
        bool firstSumPrime = true;
        bool firstSumPrimePrime = true;

        for (size_t l = 1; l < n; l++) {
            // ct_{j−1,ℓ,0} + ct_{j,ℓ,1}
            Ciphertext<DCRTPoly> leftTerm1 = (j > 0) ? maskedParts[j - 1][l - 1][0] 
                                                    : cc->EvalSub(maskedParts[j][l-1][0], maskedParts[j][l-1][0]); 
            
            Ciphertext<DCRTPoly> leftTerm2 = maskedParts[j][l - 1][1];
            
            // ct_{j−1,ℓ,2} + ct_{j,ℓ,3}
            Ciphertext<DCRTPoly> rightTerm1 = (j > 0) ? maskedParts[j - 1][l - 1][2] 
                                                    : cc->EvalSub(maskedParts[j][l-1][2], maskedParts[j][l-1][2]); 
            

            
            Ciphertext<DCRTPoly> rightTerm2 = maskedParts[j][l - 1][3];


            // ℓ-wise decrypted terms
            Plaintext pt_lterm1, pt_lterm2, pt_rterm1, pt_rterm2;
            cc->Decrypt(keys.secretKey, leftTerm1, &pt_lterm1);
            cc->Decrypt(keys.secretKey, leftTerm2, &pt_lterm2);
            cc->Decrypt(keys.secretKey, rightTerm1, &pt_rterm1);
            cc->Decrypt(keys.secretKey, rightTerm2, &pt_rterm2);

            pt_lterm1->SetLength(batchSize);
            pt_lterm2->SetLength(batchSize);
            pt_rterm1->SetLength(batchSize);
            pt_rterm2->SetLength(batchSize);

            std::cout << "j=" << j << " l=" << l << "\n";
            std::cout << "  LeftTerm1: " << pt_lterm1->GetRealPackedValue() << "\n";
            std::cout << "  LeftTerm2: " << pt_lterm2->GetRealPackedValue() << "\n";
            std::cout << "  RightTerm1: " << pt_rterm1->GetRealPackedValue() << "\n";
            std::cout << "  RightTerm2: " << pt_rterm2->GetRealPackedValue() << "\n";


            // Accumulate sums separately
            if (firstSumPrime) {
                ctCjPrime = cc->EvalAdd(leftTerm1, leftTerm2);
                firstSumPrime = false;
            } else {
                ctCjPrime = cc->EvalAdd(ctCjPrime, cc->EvalAdd(leftTerm1, leftTerm2));
            }

            if (firstSumPrimePrime) {
                ctCjPrimePrime = cc->EvalAdd(rightTerm1, rightTerm2);
                firstSumPrimePrime = false;
            } else {
                ctCjPrimePrime = cc->EvalAdd(ctCjPrimePrime, cc->EvalAdd(rightTerm1, rightTerm2));
            }
        }


        // Final composition
        auto rotated = cc->EvalRotate(ctCjPrimePrime, -static_cast<int>(n));

        /*
        Plaintext beforeRot, afterRot;
        cc->Decrypt(keys.secretKey, ctCjPrimePrime, &beforeRot);
        cc->Decrypt(keys.secretKey, rotated, &afterRot);
        beforeRot->SetLength(batchSize);
        afterRot->SetLength(batchSize);
        cout << "Before rot: " << beforeRot->GetRealPackedValue() << endl;
        cout << "After rot: " << afterRot->GetRealPackedValue() << endl;
        */
        
        ctCResults[j] = cc->EvalAdd(cc->EvalAdd(ctCResults[j], ctCjPrime), rotated);
    }

    return ctCResults;
    
}

vector<Ciphertext<DCRTPoly>> mergeCopy(vector<Ciphertext<DCRTPoly>> ctBj, size_t n, size_t c, size_t batchSize, CryptoContext<DCRTPoly> cc){
    vector<Ciphertext<DCRTPoly>> results;

    size_t ctxts = n / c; // number of ciphertexts - n is number of total diags
    
    vector<double> maskL(batchSize, 0.0);
    for (size_t ml = 0; ml < n; ml++){
        maskL[ml] = 1.0;
    }

    for (size_t j = 0; j < ctxts; j++){
        for (size_t k = 0; k < (c/2); k++){
            vector<double> maskK(batchSize, 0.0);
            for (size_t mk = 2 * k *n; mk < 2 * (k + 1) * n; mk++){
                maskK[mk] = 1.0;
            }
            auto ctjk = cc -> EvalMult(ctBj[j], cc -> MakeCKKSPackedPlaintext(maskK));

            for (size_t r = 0; r < log2(c/2); r++){
                ctjk = cc -> EvalAdd(ctjk, cc -> EvalRotate(ctjk, 2 * n * pow(2, r)));
            }
            auto intmed = cc -> EvalMult(ctjk, cc -> MakeCKKSPackedPlaintext(maskL));
            results.push_back(intmed);
            results.push_back(cc -> EvalSub(ctjk, intmed));
        }
    }


    for (size_t l = 0; l < results.size(); l++) {
        Ciphertext<DCRTPoly> replicated = results[l];

        size_t reps = batchSize / n;

        for (size_t r = 1; r < reps; r++) {
            int rot = r * n;  
            replicated = cc->EvalAdd(replicated, cc->EvalRotate(replicated, rot));
        }

        results[l] = replicated;  
    }

    return results;
}

Plaintext MakeCKKSPackedTokens(vector<double> flattenPE, CryptoContext<DCRTPoly> cc){
    return cc->MakeCKKSPackedPlaintext(flattenPE);
}

vector<Ciphertext<DCRTPoly>> extractAndReplicateDiagonals(Ciphertext<DCRTPoly> ct, size_t numDiags, size_t diagLen, size_t batchSize, CryptoContext<DCRTPoly> cc, lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys) {
    vector<Ciphertext<DCRTPoly>> replicated;

    Plaintext temp;
    cc->Decrypt(keys.secretKey, ct, &temp);
    temp->SetLength(batchSize);  // or the actual expected length
    cout << "ck in function" << temp->GetRealPackedValue() << endl;

    for (size_t d = 0; d < numDiags; d++) {
        // Step 1: Extract diagonal `d` via masking
        vector<double> maskVec(batchSize, 0.0);
        for (size_t i = 0; i < diagLen; i++) {
            maskVec[d * diagLen + i] = 1.0;
        }

        auto mask = cc->MakeCKKSPackedPlaintext(maskVec);
        auto extracted = cc->EvalMult(ct, mask);
        Ciphertext<DCRTPoly> rotated = extracted;

        // Step 2: Replicate this diagonal across the whole ciphertext
        for (size_t r = 1; r < batchSize / diagLen; r++) {
            int rot = r * diagLen;
            rotated = cc->EvalRotate(rotated, -static_cast<int>(rot));

            extracted = cc->EvalAdd(extracted, rotated);
        }

        replicated.push_back(extracted);
    }

    return replicated;
}


int main() {
    EmbeddingMatrix embeddings = {
        {0.1, 0.3, 0.2, 0.05},  // "the"
        {0.4, 0.1, 0.2, 0.3},   // "cat"
        {0.3, 0.4, 0.1, 0.2},    // "sat"
        {0.2, 0.5, 0.3, 0.4}    // "runs"
    };


    //uint32_t levelsAfter = 20;
    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;

    //uint32_t bootDepth = FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetSecurityLevel(HEStd_NotSet);        
    parameters.SetFirstModSize(60);
    parameters.SetScalingModSize(35);                      
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(25); // 30
    parameters.SetRingDim(8192);
    size_t batchSize = 16;                        
    parameters.SetBatchSize(batchSize);                         
                 

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
    
    /*
    size_t diagsQT = min(W_Q.size(), W_Q[0].size());
    vector<double> flattenIDiagQT = flattenMatrixInterlacedDiagEncoding(flattenDiagQT, rowQT, dimQT, diagsQT);
    cout << "flattenIDiagQT" << flattenIDiagQT << endl;

    */
    
    
    

    Plaintext qtptxt;
    qtptxt = cc -> MakeCKKSPackedPlaintext(flattenDiagQT);

    vector<double> decodedVector = qtptxt->GetRealPackedValue();
    cout << "qtptxt " << decodedVector << endl;

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
    Plaintext kptxt = cc -> MakeCKKSPackedPlaintext(flattenDiagK);

    size_t dimPK = dimE;
    size_t diagsPK = rowK;
    Ciphertext<DCRTPoly> ck = product(cxT, kptxt, diagsPK, dimPK, cc, keys, batchSize);
    
    Plaintext tempk;
    cc->Decrypt(keys.secretKey, ck, &tempk);
    temp->SetLength(batchSize);  // or the actual expected length
    cout << "ck" << tempk->GetRealPackedValue() << endl;
    
    
    vector<Ciphertext<DCRTPoly>> ckR = extractAndReplicateDiagonals(ck, diagsPK, dimPK, batchSize, cc, keys);
    for (size_t i = 0; i < ckR.size(); ++i) {
        Plaintext tempPlain;
        cc->Decrypt(keys.secretKey, ckR[i], &tempPlain);
        
        
        // Optional: set expected length (if needed)
        tempPlain->SetLength(batchSize);  

        // Print the real packed values
        std::cout << "ckR[" << i << "] = " << tempPlain->GetRealPackedValue() << std::endl;
    }

    vector<Ciphertext<DCRTPoly>> ctxt1 = {cqt};

    vector<Ciphertext<DCRTPoly>> p = productct(ctxt1, ckR, cc, keys, batchSize, 4, 4, 4);

    for (size_t i = 0; i < p.size(); ++i) {
        Plaintext tempP;
        cc->Decrypt(keys.secretKey, p[i], &tempP);
        
        
        // Optional: set expected length (if needed)
        tempP->SetLength(batchSize);  

        // Print the real packed values
        std::cout << "p[" << i << "] = " << tempP->GetRealPackedValue() << std::endl;
    }

    Ciphertext<DCRTPoly> pct = p[0];
    
    /*
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

*/
/*
    Ciphertext<DCRTPoly> pct2 = applySoftMax(pct, 4, cc, keys);

    Plaintext decPct2;
    cc->Decrypt(keys.secretKey, pct2, &decPct2);
    decPct2->SetLength(batchSize); // 9
    cout << "Score slots: " << decPct2->GetRealPackedValue() << endl;


*/
    
    

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

