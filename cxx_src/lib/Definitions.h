#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>


#ifndef DEFINITIONS_INCLUDED
#define DEFINITIONS_INCLUDED


#ifdef BIGNUM
typedef uint_fast64_t sz;
typedef int_fast64_t nm;
#else
typedef size_t sz;
typedef int nm;
#endif


using T = double;

using TV = std::vector<T>;
using IV = std::vector<nm>;
using IVV = std::vector<IV>;


using SpMat = Eigen::SparseMatrix<T, Eigen::RowMajor>;
using Trip = Eigen::Triplet<T>;
using VXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

#endif

