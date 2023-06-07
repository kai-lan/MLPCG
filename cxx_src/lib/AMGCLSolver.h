#ifndef AMGCL_SOLVER_H
#define AMGCL_SOLVER_H

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

#ifdef USE_MPI
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/mpi/amg.hpp>
#include <amgcl/mpi/coarsening/runtime.hpp>
#include <amgcl/mpi/make_solver.hpp>
#include <amgcl/mpi/relaxation/runtime.hpp>
#include <amgcl/solver/runtime.hpp>
//#include <amgcl/mpi/direct_solver/runtime.hpp>
//#include <amgcl/mpi/direct_solver/eigen_splu.hpp>
//#include <amgcl/mpi/direct_solver/skyline_lu.hpp>
//#include <amgcl/mpi/partition/runtime.hpp>
#include <amgcl/mpi/distributed_matrix.hpp>
#include <amgcl/mpi/preconditioner.hpp>
#include <amgcl/mpi/relaxation/as_preconditioner.hpp>
#include <amgcl/mpi/util.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
//#include <amgcl/preconditioner/dummy.hpp>
//#include <amgcl/solver/runtime.hpp>
#include <amgcl/solver/cg.hpp>
#else
#include <amgcl/amg.hpp>
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
#endif

//#include <amgcl/profiler.hpp>

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
class AMGCLSolver {
  #ifdef USE_MPI
  typedef amgcl::mpi::make_solver<amgcl::runtime::mpi::preconditioner<Backend>,
                                  amgcl::solver::cg  //<Backend>
                                                     //        amgcl::mpi::amg<
                                                     //            Backend,
                                                     //            amgcl::runtime::mpi::coarsening::wrapper<Backend>,
                                                     //            amgcl::runtime::mpi::relaxation::wrapper<Backend>,
                                                     //            //amgcl::runtime::mpi::direct::solver<double>,
                                                     //            amgcl::mpi::direct::eigen_splu<double>,
                                                     //            amgcl::runtime::mpi::partition::wrapper<Backend>
                                                     //            >,
                                                     //        amgcl::runtime::solver::wrapper
                                  >
    AMGCLSolverImpl;
  #else
  typedef amgcl::make_solver<amgcl::runtime::preconditioner<Backend>, amgcl::runtime::solver::wrapper<Backend> > AMGCLSolverImpl;
  #endif

 public:
  AMGCLSolverImpl* solver_impl;
  #ifdef USE_CUDA
  vex::Context* ctx;
  #endif

  #ifdef USE_MPI
  amgcl::mpi::communicator comm;
  // amgcl::runtime::mpi::partition::type ptype = static_cast<amgcl::runtime::mpi::partition::type>(0);
  #endif

  const SolverConfig& config;
  Backend::params bprm;
  boost::property_tree::ptree prm;
  const MPIDomain& mpi_domain;

  AMGCLSolver(const Eigen::SparseMatrix<T, Eigen::RowMajor, nm>& A_eigen_reduced, const SolverConfig& _config, const MPIDomain& _mpi_domain);

  ~AMGCLSolver();

  // recompute pc
  void Solve(const Eigen::SparseMatrix<T, Eigen::RowMajor, nm>& A_eigen_reduced, Eigen::Matrix<T, Eigen::Dynamic, 1>& x, const Eigen::Matrix<T, Eigen::Dynamic, 1>& b);

  // reuse pc
  void Solve(Eigen::Matrix<T, Eigen::Dynamic, 1>& x, const Eigen::Matrix<T, Eigen::Dynamic, 1>& b);

 private:
  void setParams();

  int n;
};


#endif

