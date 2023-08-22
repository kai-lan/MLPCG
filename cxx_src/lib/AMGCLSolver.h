#ifndef AMGCL_SOLVER_H
#define AMGCL_SOLVER_H

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

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

#ifndef USE_VEXCEL
  #ifndef USE_CUDA
    AMGCL_USE_EIGEN_VECTORS_WITH_BUILTIN_BACKEND()
  #endif
#endif

// Need a stateful solver object so we can reuse the AMGCL preconditioner between solves
template <typename PBackend, typename SBackend>
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
  const SolverConfig& config;
  typename SBackend::params bprm;
  boost::property_tree::ptree prm;

  #ifdef USE_VEXCL
    vex::Context* ctx;
  #endif


  AMGCLSolver(const SpMat& A, const SolverConfig& _config) : config(_config) {
    #ifdef USE_VEXCL
      std::cout << "Using VexCL" << std::endl;
      ctx = new vex::Context(vex::Filter::Exclusive(vex::Filter::GPU && vex::Filter::Env && vex::Filter::Count(1)));  // vex::Filter::DoublePrecision && vex::Filter::Count(1)));
    #elif USE_CUDA
      std::cout << "Using CUDA" << std::endl;
      cusparseCreate(&bprm.cusparse_handle);
    #else
      std::cout << "Using OpenMP" << std::endl;
    #endif
    #ifdef MIXED_PRECISION
      std::cout << "Using mixed precesion" << std::endl;
    #endif
    setParams();

    n = A.rows();
    solver = std::make_unique<Solver>(A, prm, bprm);
  }

  AMGCLSolver(const SolverConfig& _config) : config(_config) {
  #ifdef USE_VEXCL
    // std::cout << "Using VexCL" << std::endl;
    ctx = new vex::Context(vex::Filter::Exclusive(vex::Filter::GPU && vex::Filter::Env && vex::Filter::Count(1)));  // vex::Filter::DoublePrecision && vex::Filter::Count(1)));
  #elif USE_CUDA
    // std::cout << "Using CUDA" << std::endl;
    cusparseCreate(&bprm.cusparse_handle);
  #else
    // std::cout << "Using OpenMP" << std::endl;
  #endif
  #ifdef MIXED_PRECISION
    // std::cout << "Using mixed precesion" << std::endl;
  #endif
    setParams();
  }

  ~AMGCLSolver() {
  #ifdef USE_VEXCL
    if (ctx)
      delete ctx;
  #endif
  }

  // recompute pc
  std::tuple<int, T> Solve(const SpMat& A, VXT& x, const VXT& b, bool profile=false) {
    if (profile) prof.tic("setup");
    solver = std::make_unique<Solver>(A, prm, bprm);
    if (profile) prof.toc("setup");
    if (profile) prof.tic("solve");
    auto info = Solve(x, b, profile);
    if (profile) prof.toc("solve");
    if (profile)
      std::cout << prof << std::endl;
    return info;
  }
  // reuse pc
  std::tuple<int, T> Solve(VXT& x, const VXT& b, bool profile=false) {
    sz iters;
    T error;
  #ifdef USE_VEXCL
    vex::vector<T> b_vexcl(*ctx, b.size(), b.data());
    vex::vector<T> x_vexcl(*ctx, x.size(), x.data());
    std::tie(iters, error) = solver->operator()(b_vexcl, x_vexcl);
    vex::copy(x_vexcl.begin(), x_vexcl.end(), x.data());
  #elif USE_CUDA
    thrust::device_vector<T> b_cuda(b.data(), b.data() + b.size());
    thrust::device_vector<T> x_cuda(x.data(), x.data() + x.size());
    std::tie(iters, error) = solver->operator()(b_cuda, x_cuda);
    thrust::copy(x_cuda.begin(), x_cuda.end(), x.begin());
  #else
    if (profile) prof.tic("solve");
    std::tie(iters, error) = solver->operator()(b, x);
    if (profile) prof.toc("solve");
  #endif
    return std::make_tuple (iters, error);
  }

 private:
  void setParams() {
    // std::cout << "Tol: " << config.tol << std::endl;
    // std::cout << "Max iters: " << config.max_iter << std::endl;
  #ifdef USE_VEXCL
    bprm.q = *ctx;
  #endif
    prm.put("solver.tol", config.tol);
    prm.put("solver.maxiter", config.max_iter);
  }

  int n;
};

#endif
