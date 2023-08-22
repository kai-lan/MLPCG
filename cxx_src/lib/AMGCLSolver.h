#ifndef AMGCL_SOLVER_H
#define AMGCL_SOLVER_H

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#ifdef USE_VEXCL
  #include <amgcl/backend/vexcl.hpp>
  typedef amgcl::backend::vexcl<double> SBackend;
  #ifdef MIXED_PRECISION
    typedef amgcl::backend::vexcl<float> PBackend;
  #else
    typedef amgcl::backend::vexcl<double> PBackend;
  #endif
#elif USE_CUDA
  #include <amgcl/backend/cuda.hpp>
  typedef amgcl::backend::cuda<double> SBackend;
  #ifdef MIXED_PRECISION
    typedef amgcl::backend::cuda<float> PBackend;
  #else
    typedef amgcl::backend::cuda<double> PBackend;
  #endif
#else
  #include <amgcl/backend/builtin.hpp>
  typedef amgcl::backend::builtin<double> SBackend;
  #ifdef MIXED_PRECISION
    typedef amgcl::backend::builtin<float> PBackend;
  #else
    typedef amgcl::backend::builtin<double> PBackend;
  #endif
#endif

#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
// #include <amgcl/solver/bicgstab.hpp>

#include <amgcl/adapter/eigen.hpp>
#include <amgcl/adapter/reorder.hpp>

#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

#include <SolverConfig.h>

#ifndef USE_VEXCEL
  #ifndef USE_CUDA
    AMGCL_USE_EIGEN_VECTORS_WITH_BUILTIN_BACKEND()
  #endif
#endif

// Need a stateful solver object so we can reuse the AMGCL preconditioner between solves
class AMGCLSolver {

  typedef amgcl::make_solver<
    amgcl::amg<
      PBackend,
      amgcl::coarsening::smoothed_aggregation,
      amgcl::relaxation::spai0
      >,
    amgcl::solver::cg<SBackend>
    > Solver;

 public:
  std::unique_ptr<Solver> solver;
  amgcl::profiler<> prof{"AMGCL solve"};

  #ifdef USE_VEXCL
    vex::Context* ctx;
  #endif

  const SolverConfig& config;
  SBackend::params bprm;
  boost::property_tree::ptree prm;

  AMGCLSolver(const SpMat& A, const SolverConfig& config);
  AMGCLSolver(const SolverConfig& config);

  ~AMGCLSolver();

  // recompute pc
  std::tuple<int, T> Solve(const SpMat& A, VXT& x, const VXT& b, bool profile=false);
  // reuse pc
  std::tuple<int, T> Solve(VXT& x, const VXT& b, bool profile=false);

 private:
  void setParams();

  int n;
};

#include "AMGCLSolver.inl"

#endif
