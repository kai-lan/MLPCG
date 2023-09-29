#ifndef ICSOLVER
#define ICSOLVER

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <amgcl/make_solver.hpp>
// #include <amgcl/amg.hpp>
// #include <amgcl/coarsening/smoothed_aggregation.hpp>
// #include <amgcl/relaxation/spai0.hpp>

#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/solver/cg.hpp>
// #include <amgcl/solver/bicgstab.hpp>

#include <amgcl/adapter/eigen.hpp>
#include <amgcl/adapter/reorder.hpp>

#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

#include <Definitions.h>

#ifdef USE_VEXCL
  #include <amgcl/backend/vexcl.hpp>
  #include <amgcl/relaxation/ilu0.hpp>
  typedef amgcl::backend::vexcl<double> SBackend;
  #ifdef MIXED_PRECISION
    typedef amgcl::backend::vexcl<float> PBackend;
  #else
    typedef amgcl::backend::vexcl<double> PBackend;
  #endif
#elif USE_CUDA
  #include <amgcl/backend/cuda.hpp>
  #include <amgcl/relaxation/cusparse_ilu0.hpp>
  typedef amgcl::backend::cuda<double> SBackend;
  #ifdef MIXED_PRECISION
    typedef amgcl::backend::cuda<float> PBackend;
  #else
    typedef amgcl::backend::cuda<double> PBackend;
  #endif
#else
  #include <amgcl/backend/builtin.hpp>
  #include <amgcl/relaxation/ilu0.hpp>
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
struct ICSolver {

 public:
  typedef amgcl::make_solver<
    amgcl::relaxation::as_preconditioner<
      PBackend,
      amgcl::relaxation::ilu0
      >,
    amgcl::solver::cg<SBackend>
    > Solver;


  static std::tuple<int, T, T, T> Solve(const SpMat& A, VXT& x, const VXT& b, double tol=1e-4, double atol=1e-10, int max_iters=100, bool verbose=false) {

    typename Solver::params prm;
    prm.solver.tol = std::max(tol, atol / b.norm());
    prm.solver.maxiter = max_iters;
    prm.solver.verbose = verbose;
    typename SBackend::params bprm;
  #ifdef USE_VEXCL
    vex::Context ctx(vex::Filter::Exclusive(vex::Filter::GPU && vex::Filter::Env && vex::Filter::Count(1)));
    bprm.q = ctx;
  #elif USE_CUDA
    cusparseCreate(&bprm.cusparse_handle);
  #endif

    amgcl::profiler<> prof("IC solver");
    T setup_time, solve_time;

    prof.tic("setup");
    Solver solver(A, prm, bprm);
    setup_time = prof.toc("setup");

    int iters;
    T error;
  #ifdef USE_VEXCL
    prof.tic("Transferring data");
    vex::vector<T> b_vexcl(ctx, b.size(), b.data());
    vex::vector<T> x_vexcl(ctx, x.size(), x.data());
    prof.toc("Transferring data");

    prof.tic("solve");
    std::tie(iters, error) = solver(b_vexcl, x_vexcl);
    solve_time = prof.toc("solve");

    vex::copy(x_vexcl.begin(), x_vexcl.end(), x.data());
  #elif USE_CUDA
    prof.tic("Transferring data");
    thrust::device_vector<T> b_cuda(b.data(), b.data() + b.size());
    thrust::device_vector<T> x_cuda(x.data(), x.data() + x.size());
    prof.toc("Transferring data");

    prof.tic("solve");
    std::tie(iters, error) = solver(b_cuda, x_cuda);
    solve_time = prof.toc("solve");

    thrust::copy(x_cuda.begin(), x_cuda.end(), x.begin());
  #else
    prof.tic("solve");
    std::tie(iters, error) = solver(b, x);
    solve_time = prof.toc("solve");
  #endif

    std::cout << solver << std::endl;
    std::cout << prof << std::endl;

    return std::make_tuple(iters, setup_time, solve_time, error);
  }

};


#endif /* ICSOLVER */
