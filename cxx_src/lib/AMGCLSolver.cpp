#include "AMGCLSolver.h"


AMGCLSolver::AMGCLSolver(const SpMat& A, const SolverConfig& _config) : config(_config) {
  #ifdef USE_VEXCL
    std::cout << "Using VexCL" << std::endl;
    ctx = new vex::Context(vex::Filter::Exclusive(vex::Filter::GPU && vex::Filter::Env && vex::Filter::Count(1)));  // vex::Filter::DoublePrecision && vex::Filter::Count(1)));
  #elif USE_CUDA
    std::cout << "Using CUDA" << std::endl;
    cusparseCreate(&bprm.cusparse_handle);
  #endif
  setParams();

  n = A.rows();
  solver = std::make_unique<Solver>(A, prm, bprm);
}

AMGCLSolver::AMGCLSolver(const SolverConfig& _config) : config(_config) {
  #ifdef USE_VEXCL
    std::cout << "Using VexCL" << std::endl;
    ctx = new vex::Context(vex::Filter::Exclusive(vex::Filter::GPU && vex::Filter::Env && vex::Filter::Count(1)));  // vex::Filter::DoublePrecision && vex::Filter::Count(1)));
  #endif
  setParams();
}

AMGCLSolver::~AMGCLSolver() {
  #ifdef USE_VEXCL
    if (ctx)
      delete ctx;
  #endif
}

void AMGCLSolver::Solve(const SpMat& A, VXT& x, const VXT& b, bool profile) {
  if (profile) prof.tic("setup");
  solver = std::make_unique<Solver>(A, prm, bprm);
  if (profile) prof.toc("setup");
  if (profile) prof.tic("solve");
  Solve(x, b, profile);
  if (profile) prof.toc("solve");
  if (profile)
    std::cout << prof << std::endl;
}

void AMGCLSolver::Solve(VXT& x, const VXT& b, bool profile) {
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
  #else
    if (profile) prof.tic("solve");
    std::tie(iters, error) = solver->operator()(b, x);
    if (profile) prof.toc("solve");
  #endif
}

void AMGCLSolver::setParams() {
  std::cout << "Tol: " << config.tol << std::endl;
  std::cout << "Max iters: " << config.max_iter << std::endl;

  #ifdef USE_VEXCL
    bprm.q = *ctx;
  #endif

  prm.put("solver.tol", config.tol);
  prm.put("solver.maxiter", config.max_iter);
  // prm.put("precond.class", "amg");
  // prm.put("precond.relax.type", "chebyshev");
  // prm.put("precond.relax.degree", 2);
  // prm.put("precond.relax.power_iters", 100);
  // prm.put("precond.relax.higher", 2.0f);
  // prm.put("precond.relax.lower", 1.0f / 120.0f);
  // prm.put("precond.relax.scale", true);
  // prm.put("precond.max_levels", 6);

  // prm.put("precond.direct_coarse", false);
  // prm.put("solver.type", "cg");

  // prm.put("precond.coarsening.type", "smoothed_aggregation");
  // prm.put("precond.coarsening.estimate_spectral_radius", true);
  // prm.put("precond.coarsening.relax", 1.0f);
  // prm.put("precond.coarsening.power_iters", 100);
  // prm.put("precond.coarse_enough", 5000);
  // prm.put("precond.ncycle", 2);

  /*prm.put("solver.type", config.solver);
  prm.put("solver.tol", config.tol);
  prm.put("solver.maxiter", config.max_iter);
  if (config.amgcl_precond_coarsening_type != "")
      prm.put("precond.coarsening.type", config.amgcl_precond_coarsening_type);
  else
      prm.put("precond.coarsening.type", "smoothed_aggregation");
  if (config.amgcl_precond_relaxation_type != "")
      prm.put("precond.relaxation.type", config.amgcl_precond_relaxation_type);
  else
      prm.put("precond.relaxation.type", "chebyshev");

  if (config.amgcl_precond_coarse_enough != T(-1))
      prm.put("precond.coarse_enough", config.amgcl_precond_coarse_enough);

  if (config.amgcl_precond_max_levels != -1)
      prm.put("precond.max_levels", config.amgcl_precond_max_levels);

  if (config.amgcl_precond_direct_coarse)
      prm.put("precond.direct_coarse", false);

  if (config.amgcl_solver_direct_coarse)
     prm.put("solver.direct_coarse", false);

  if (config.amgcl_precond_class != "")
      prm.put("precond.class", config.amgcl_precond_class);

  if (config.amgcl_cg_project)
      prm.put("solver.project_out_constant_nullspace", true);*/
}



