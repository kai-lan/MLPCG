#define EIGEN_NO_CUDA
#define SOLVER_DEBUG

#include "NoMPIAMGCLSolver.h"
#include <chrono>
#include <Definitions.h>

#ifndef USE_CUDA
AMGCL_USE_EIGEN_VECTORS_WITH_BUILTIN_BACKEND()
#endif


NoMPIAMGCLSolver::NoMPIAMGCLSolver(const Eigen::SparseMatrix<T, Eigen::RowMajor, nm>& A_eigen_reduced, const SolverConfig& _config, const MPIDomain& _mpi_domain)
    : config(_config), mpi_domain(_mpi_domain) {
  #ifdef USE_CUDA
  std::cout<<"CUDA Using Mode"<<std::endl; 
  #ifdef SOLVER_DEBUG
  // std::cout<<"Creating VexCL context"<<std::endl;
  #endif
  ctx = new vex::Context(vex::Filter::Exclusive(vex::Filter::GPU && vex::Filter::Env && vex::Filter::Count(1)));  // vex::Filter::DoublePrecision && vex::Filter::Count(1)));
  #ifdef SOLVER_DEBUG
  // std::cout<<"Context created: "<<*ctx<<std::endl;
  #endif
  #endif

  prof = new amgcl::profiler<>("poisson3Db");
  setParams();

  n = A_eigen_reduced.rows();

  #ifdef SOLVER_DEBUG
  auto t0 = std::chrono::high_resolution_clock::now();
  #endif
  prof->tic("setup");
  // Use AMGCL's Eigen adapter to get the matrix into their internal format
  solver_impl = new NoMPIAMGCLSolverImpl(A_eigen_reduced, prm, bprm);
  prof->toc("setup");
  #ifdef SOLVER_DEBUG
  auto t1 = std::chrono::high_resolution_clock::now();
  printf("Built AMGCL solve object and preconditioner in %f seconds\n", (float)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.);
  #endif
}

NoMPIAMGCLSolver::~NoMPIAMGCLSolver() {
  if (solver_impl)
    delete solver_impl;
  #ifdef USE_CUDA
  if (ctx)
    delete ctx;
  #endif
}

void NoMPIAMGCLSolver::Solve(const Eigen::SparseMatrix<T, Eigen::RowMajor, nm>& A_eigen_reduced, Eigen::Matrix<T, Eigen::Dynamic, 1>& x, const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) {
  if (solver_impl)
    delete solver_impl;

  n = A_eigen_reduced.rows();

  // Use AMGCL's Eigen adapter to get the matrix into their internal format
  solver_impl = new NoMPIAMGCLSolverImpl(A_eigen_reduced, prm, bprm);
  #ifdef SOLVER_DEBUG
  printf("Made solve obj...\n");
  #endif
  Solve(x, b);
}

void NoMPIAMGCLSolver::Solve(Eigen::Matrix<T, Eigen::Dynamic, 1>& x, const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) {
  Eigen::VectorXd _b = config.amgcl_rhs_scaling * b;
  printf("NoMPI AMGCL Solver initial residual / norm _b is %f\n", _b.norm());
  #ifdef USE_CUDA
  // Create device vector for b
  vex::vector<T> b_vexcl(n, _b.data());
  #ifdef SOLVER_DEBUG
  printf("Done with that...\n");
  #endif
  #endif
  //Eigen::Matrix<T, Eigen::Dynamic, 1> xr = Eigen::MatrixXd::Random(n, 1);
  Eigen::Matrix<T, Eigen::Dynamic, 1> xr = x;//Eigen::MatrixXd::Random(n, 1);
  #ifdef USE_CUDA
  vex::vector<T> x_vexcl(n, xr.data());
  #endif
  size_t iters;
  double error;
  #ifdef SOLVER_DEBUG
  auto t0 = std::chrono::high_resolution_clock::now();
  printf("copied vecs to device...\n");
  #endif
  #ifdef USE_CUDA
  prof->tic("solve");
  std::tie(iters, error) = solver_impl->operator()(b_vexcl, x_vexcl);
  prof->toc("solve");
  #else
  prof->tic("solve");
  std::tie(iters, error) = solver_impl->operator()(_b, xr);
  prof->toc("solve");

  #endif
  #ifdef SOLVER_DEBUG
  auto t1 = std::chrono::high_resolution_clock::now();
  printf("AMGCL solve done: %f seconds\n iters = %d\n error = %f\n", (float)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000., (int)iters, error);
  #endif
  #ifdef USE_CUDA
  // Transfer solution back from device to host
  for (size_t i = 0; i < x_vexcl.size(); ++i)
    x[i] = x_vexcl[i];
  #else
  for (nm i = 0; i < xr.size(); ++i)
    x[i] = xr[i];
  #endif
  x /= config.amgcl_rhs_scaling;
  std::cout << *solver_impl << std::endl;

  std::cout << "Iters: " << iters << std::endl
              << "Error: " << error << std::endl
              << *prof << std::endl;
}

void NoMPIAMGCLSolver::setParams() {
  std::cout << "CFG TOL = " << config.tol << std::endl;
  std::cout << "CFG MAX ITER = " << config.max_iter << std::endl;

  #ifdef USE_CUDA
  bprm.q = *ctx;
  #endif

  //prm.put("solver.type", "bicgstab");

  prm.put("solver.type", "cg");
  prm.put("solver.verbose", true);
  prm.put("solver.abstol", config.tol);
  prm.put("solver.maxiter", config.max_iter);
  prm.put("solver.ns_search", true);
  
  prm.put("precond.class", "amg");
  //prm.put("precond.class", "relaxation");
  prm.put("precond.relax.type", "chebyshev");
  prm.put("precond.ncycle", 2);
  //prm.put("precond.direct_coarse", true);
  prm.put("precond.coarsening.type", "aggregation");
  //prm.put("precond.coarsening.type", "smoothed_aggregation");

  prm.put("solver.verbose", true);
  /*
  prm.put("solver.verbose", true);
  prm.put("solver.type", "preonly");
  prm.put("solver.verbose", true);
  //prm.put("precond.relax.type", "chebyshev");
  prm.put("precond.relax.type", "damped_jacobi");

  */

  //prm.put("precond.class", "relaxation");
  //prm.put("precond.type", "damped_jacobi");
  //prm.put("precond.type", "chebyshev");
  //prm.put("precond.type", "ilut");

  /*
  prm.put("solver.type", "preonly");
  prm.put("solver.verbose", true);
  prm.put("precond.class", "relaxation");
  //prm.put("precond.type", "chebyshev");
  prm.put("precond.type", "damped_jacobi");
  */
  //prm.put("precond.direct_coarse", true);
  //prm.put("precond.direct_coarse", false);//if make it true, ilu is not worked because diagonal is zeros
  //prm.put("solver.type", "cg");
  //prm.put("precond.relax.type", "as_preconditioner");
  //prm.put("solver.type", "cg");
  //prm.put("solver.ns_search", true);
  //prm.put("precond.relax.type", "spai0");
  //prm.put("precond.coarsening.type", "smoothed_aggregation");
  //prm.put("solver.type", "cg");
  //prm.put("precond.relax.type", "damped_jacobi");
  //prm.put("precond.relax.scale", true);
  //prm.put("precond.ncycle", 2);
  
  //prm.put("precond.relax.degree", 2);
  //prm.put("precond.relax.power_iters", 100);
  //prm.put("precond.relax.higher", 2.0f);
  //prm.put("precond.relax.lower", 1.0f / 120.0f);
  //prm.put("precond.relax.scale", true);
  //prm.put("precond.max_levels", 6);

  //prm.put("solver.type", "cg");

  //prm.put("precond.coarsening.estimate_spectral_radius", true);
  //prm.put("precond.coarsening.relax", 1.0f);
  //prm.put("precond.coarsening.power_iters", 100);
  //prm.put("precond.coarse_enough", 5000);
  //prm.put("precond.ncycle", 2);
}


