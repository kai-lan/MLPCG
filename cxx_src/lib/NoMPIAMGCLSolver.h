#ifndef NO_MPI_AMGCL_SOLVER_H
#define NO_MPI_AMGCL_SOLVER_H

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#ifdef USE_CUDA
#include <amgcl/backend/vexcl.hpp>
//#include <amgcl/backend/cuda.hpp>
//#include <amgcl/relaxation/cusparse_ilu0.hpp>
typedef amgcl::backend::vexcl<double> Backend;
#else
#include <amgcl/backend/builtin.hpp>
typedef amgcl::backend::builtin<double> Backend;
#endif

//#include <amgcl/amg.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/solver/runtime.hpp>
//#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/adapter/reorder.hpp>
//#include <amgcl/io/mm.hpp>
//#include <amgcl/io/binary.hpp>

#include <amgcl/profiler.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/solver/preonly.hpp>
#include <Eigen/Sparse>

#include <SolverConfig.h>
#include <MPIDomain.h>


#ifdef BIGNUM
typedef uint_fast64_t sz;
typedef int_fast64_t nm;
#else
typedef size_t sz;
typedef int nm;
#endif


// Need a stateful solver object so we can reuse the AMGCL preconditioner between solves
class NoMPIAMGCLSolver {
  //typedef amgcl::make_solver<amgcl::runtime::preconditioner<Backend>, amgcl::runtime::solver::wrapper<Backend> > NoMPIAMGCLSolverImpl;
  typedef amgcl::make_solver<amgcl::runtime::preconditioner<Backend>, amgcl::runtime::solver::wrapper<Backend> > NoMPIAMGCLSolverImpl;

 public:
  NoMPIAMGCLSolverImpl* solver_impl;
  #ifdef USE_CUDA
  vex::Context* ctx;
  #endif

  const SolverConfig& config;
  Backend::params bprm;
  boost::property_tree::ptree prm;
  const MPIDomain& mpi_domain;

  amgcl::profiler<> *prof;

  NoMPIAMGCLSolver(const Eigen::SparseMatrix<T, Eigen::RowMajor, nm>& A_eigen_reduced, const SolverConfig& _config, const MPIDomain& _mpi_domain);

  ~NoMPIAMGCLSolver();

  // recompute pc
  void Solve(const Eigen::SparseMatrix<T, Eigen::RowMajor, nm>& A_eigen_reduced, Eigen::Matrix<T, Eigen::Dynamic, 1>& x, const Eigen::Matrix<T, Eigen::Dynamic, 1>& b);

  // reuse pc
  void Solve(Eigen::Matrix<T, Eigen::Dynamic, 1>& x, const Eigen::Matrix<T, Eigen::Dynamic, 1>& b);

 private:
  void setParams();

  int n;
};


#endif
