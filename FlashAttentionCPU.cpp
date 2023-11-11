#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
using namespace std;

// define precision
using InputPrec=float;
using CompPrec=double;

constexpr size_t N = 16;
constexpr size_t d = 2;
constexpr size_t M = 16; // SRAM size
constexpr size_t Bc = M / (4*d);
constexpr size_t Br = std::min(Bc, d);
constexpr size_t Tr = N / Br;
constexpr size_t Tc = N / Bc;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dist(0.0, 1.0);

template<class T>
void InitInput(vector<vector<T>> &MatQ, vector<vector<T>> &MatK, vector<vector<T>> &MatV) {
    for (size_t j = 0; j < d; ++j)
    for (size_t i = 0; i < N; ++i) {
        MatQ[i][j] = dist(gen);
        MatK[j][i] = dist(gen);
        MatV[i][j] = dist(gen);
    }
}

template<class T>
void LoadBlockKV(vector<vector<T>> &SubMatK, vector<vector<T>> &SubMatV, 
        vector<vector<T>> &MatK, vector<vector<T>> &MatV, const size_t j0) {
    for (size_t j = 0; j < Bc; ++j) {
        auto j1 = j0 * Bc + j;
        for (size_t i = 0; i < d; ++i) {
            SubMatK[i][j] = MatK[i][j1];
            SubMatV[j][i] = MatV[j1][i];
        }
    }
}

template<class T>
void LoadBlockQO(vector<vector<T>> &SubMatQ, vector<vector<T>> &SubMatO, 
        vector<vector<T>> &MatQ, vector<vector<T>> &MatO,
        vector<T> &SubVecL, vector<T> &SubVecM, 
        vector<T> &VecL, vector<T> &VecM, const size_t i0) {
    for (size_t i = 0; i < Br; ++i) {
        auto i1 = i0 * Br + i;
        SubVecL[i] = VecL[i1];
        SubVecM[i] = VecM[i1];
        for (size_t j = 0; j < d; ++j) {
            SubMatQ[i][j] = MatQ[i1][j];
            SubMatO[i][j] = MatO[i1][j];
        }
    }
}

template<class T>
void Matmul(vector<vector<T>> &C, vector<vector<T>> &A, vector<vector<T>> &B) {
    for (size_t j = 0; j < B[0].size(); ++j) {
        for (size_t i = 0; i < A.size(); ++i) {
            C[i][j] = 0.0;
            for (size_t k = 0; k < A[0].size(); ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

template<class T>
void Softmax(vector<T> &SubVecL, vector<vector<T>> &SubMatS, vector<T> &SubVecM, 
        vector<T> &LocM, vector<T> &LocL, vector<T> &NewLocM, vector<T> &NewLocL, 
        vector<vector<T>> &SubMatP) {
    // obtain the M.
    for (size_t i = 0; i < Br; ++i) {
        LocM[i] = 0.0;
        for (size_t j = 0; j < Bc; ++j) {
            if (LocM[i] < SubMatS[i][j] ) {
                LocM[i] = SubMatS[i][j];
            }
        }
        if (LocM[i] > SubVecM[i]) {
            NewLocM[i] = LocM[i];
        }
        else {
            NewLocM[i] = SubVecM[i];
        }
    }
    //
    for (size_t j = 0; j < Bc; ++j) {
        for (size_t i = 0; i < Br; ++i) {
            SubMatP[i][j] = std::exp(SubMatS[i][j] - LocM[i]);
        }
    }
}

template<class T>
void GlobalSoftmax(vector<vector<T>> &MatP_ref, vector<vector<T>> &MatS_ref) {
    vector<T> Gm(N, 0.0);
    vector<T> Gl(N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        Gm[i] = 0.0;
        for (size_t j = 0; j < N; ++j) {
            if (MatS_ref[i][j] > Gm[i]) Gm[i] = MatS_ref[i][j];
        }
        Gl[i] = 0.0;
        for (size_t j = 0; j < N; ++j) {
            MatP_ref[i][j] = std::exp(MatS_ref[i][j]-Gm[i]);
            Gl[i] += MatP_ref[i][j];
        }
        for (size_t j = 0; j < N; ++j) {
            MatP_ref[i][j] /= Gl[i];
        }
    }
}
int main (int argc, char **argv)
{
    vector<vector<InputPrec>> MatQ(N,vector<InputPrec>(d)); // on HBM
    vector<vector<InputPrec>> MatK(d,vector<InputPrec>(N)); // on HBM
    vector<vector<InputPrec>> MatV(N,vector<InputPrec>(d)); // on HBM
    vector<vector<InputPrec>> MatO(N,vector<InputPrec>(d,0.0)); // on HBM
    vector<vector<InputPrec>> MatS(N, vector<InputPrec>(N)); // on HBM
    vector<InputPrec> VecL(N, 0.0); // on HBM
    vector<InputPrec> VecM(N, -100000.0); // on HBM
    //define the block sub-matrices.
    vector<vector<InputPrec>> SubMatQ(Br, vector<InputPrec>(d));
    vector<vector<InputPrec>> SubMatK(d, vector<InputPrec>(Bc));
    vector<vector<InputPrec>> SubMatV(Bc, vector<InputPrec>(d));
    vector<vector<InputPrec>> SubMatO(Br, vector<InputPrec>(d));
    vector<vector<InputPrec>> SubMatS(Br, vector<InputPrec>(Bc));
    vector<InputPrec> SubVecL(Br);
    vector<InputPrec> SubVecM(Br);
    //
    vector<InputPrec> LocM(Br, 0.0);
    vector<InputPrec> LocL(Br, 0.0);
    vector<InputPrec> NewLocM(Br, 0.0);
    vector<InputPrec> NewLocL(Br, 0.0);
    vector<vector<InputPrec>> SubMatP(Br, vector<InputPrec>(Bc));
    vector<vector<InputPrec>> TempO(Br, vector<InputPrec>(d));
    //
    std::cout << "N=" << N << ", ";
    std::cout << "d=" << d << ", ";
    std::cout << "M=" << M << ", ";
    std::cout << "Bc=" << Bc << ", ";
    std::cout << "Br=" << Br << ", ";
    std::cout << "Tc=" << Tc << ", ";
    std::cout << "Tr=" << Tr << ", " << std::endl;
    // initilize Q, K, V
    InitInput(MatQ, MatK, MatV);
    //
    for (size_t j = 0; j < Tc; ++j) {
       // load SubMatK, SubMatV from HBM to SRAM.
       std::cout << "Outer Iteration: j = " << j << std::endl;
       LoadBlockKV(SubMatK, SubMatV, MatK, MatV, j);
       //std::cout << "Load Block K and V Succeed!" << std::endl;
       for (size_t i = 0; i < Tr; ++i) {
           // load SubMatQ, SubMatO, SubVecL, SubVecM from HBM to SRAM.
           LoadBlockQO(SubMatQ, SubMatO, MatQ, MatO, SubVecL, SubVecM, VecL, VecM, i);
           // On-chip: compute S=QK^T(Br*Bc).
           Matmul(SubMatS, SubMatQ, SubMatK);
           // on-chip, compute m_ij = rowmax(S_ij)(Br*1), P_ij = exp(S_ij - m_ij)(Br*Bc), 
           // L_ij=rowsum(P_ij)(Br*1).
           Softmax(SubVecL, SubMatS, SubVecM, LocM, LocL, NewLocM, NewLocL, SubMatP);
           //
           // on chip.
           for (size_t i1 = 0; i1 < Br; ++i1) {
               LocL[i1] = 0.0;
               for (size_t j1 = 0; j1 < Bc; ++j1) {
                   LocL[i1] += SubMatP[i1][j1];
               }
               if ( j == 0 ) {
                   NewLocL[i1] = std::exp(LocM[i1]-NewLocM[i1]) * LocL[i1];
               }
               else {
                   NewLocL[i1] = std::exp(SubVecM[i1]-NewLocM[i1]) * SubVecL[i1];
                   NewLocL[i1] += std::exp(LocM[i1]-NewLocM[i1]) * LocL[i1];
               }
           }
           // obtain PV
           Matmul(TempO, SubMatP, SubMatV);
           // update O and L, M
           for (size_t i1 = 0; i1 < Br; ++i1) {
               for (size_t j1 = 0; j1 < d; ++j1) {
                   if (j == 0) {
                       SubMatO[i1][j1] = (std::exp(LocM[i1]-NewLocM[i1]) * TempO[i1][j1]) / NewLocL[i1];
                   }
                   else {
                       SubMatO[i1][j1] = ( SubVecL[i1] * std::exp(SubVecM[i1]-NewLocM[i1]) * SubMatO[i1][j1]
                           + std::exp(LocM[i1]-NewLocM[i1]) * TempO[i1][j1] ) / NewLocL[i1];
                   }
                   MatO[i1+i*Br][j1] = SubMatO[i1][j1];
               }
           }
           // update SubVecL, SubVecM on HBM
           for (size_t i1 = 0; i1 < Br; ++i1) {
               SubVecM[i1] = NewLocM[i1];
               SubVecL[i1] = NewLocL[i1];
               VecM[i1+Br*i] = SubVecM[i1];
               VecL[i1+Br*i] = SubVecL[i1];
           }
       }
    }
    //
    // standard Attention Matrix.
    vector<vector<InputPrec>> MatO_ref(N,vector<InputPrec>(d));
    vector<vector<InputPrec>> MatS_ref(N, vector<InputPrec>(N));
    vector<vector<InputPrec>> MatP_ref(N, vector<InputPrec>(N));
    //
    Matmul(MatS_ref, MatQ, MatK);
    //
    GlobalSoftmax(MatP_ref, MatS_ref);
    //
    Matmul(MatO_ref, MatP_ref, MatV);
    //
    // verify results.
    for (size_t j = 0; j < d; ++j) {
        for (size_t i = 0; i < N; ++i) {
            if (std::abs(MatO_ref[i][j] - MatO[i][j]) > 1.0e-4) {
                std::cout << "(i,j) = " << i << ", " << j << std::endl;
                std::cout << "Wrong Results!" << std::endl;
                return 0;
            }
        }
    }
    std::cout << "Pass!" << std::endl;
    return 0;
}
