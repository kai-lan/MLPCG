#define EIGEN_NO_CUDA
#define SOLVER_DEBUG

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <chrono>

#include <Definitions.h>

#include "AMGCLSolver.h"

#ifndef USE_CUDA
AMGCL_USE_EIGEN_VECTORS_WITH_BUILTIN_BACKEND()
#endif

AMGCLSolver::AMGCLSolver(const Eigen::SparseMatrix<T, Eigen::RowMajor, nm>& A_eigen_reduced, const SolverConfig& _config, const MPIDomain& _mpi_domain) : config(_config), mpi_domain(_mpi_domain) {
  #ifdef USE_MPI
  comm = amgcl::mpi::communicator(MPI_COMM_WORLD);
  #endif

  #ifdef USE_CUDA
  #ifdef SOLVER_DEBUG
  // std::cout<<"Creating VexCL context"<<std::endl;
  #endif
  ctx = new vex::Context(vex::Filter::Exclusive(vex::Filter::GPU && vex::Filter::Env && vex::Filter::Count(1)));  // vex::Filter::DoublePrecision && vex::Filter::Count(1)));
  #ifdef SOLVER_DEBUG
  // std::cout<<"Context created: "<<*ctx<<std::endl;
  #endif
  #endif

  setParams();

  n = A_eigen_reduced.rows();

  #ifdef SOLVER_DEBUG
  auto t0 = std::chrono::high_resolution_clock::now();
  #endif
  // Use AMGCL's Eigen adapter to get the matrix into their internal format
  #ifdef USE_MPI
  solver_impl = new AMGCLSolverImpl(comm, A_eigen_reduced, prm, bprm);
  #else
  solver_impl = new AMGCLSolverImpl(A_eigen_reduced, prm, bprm);
  #endif
  #ifdef SOLVER_DEBUG
  auto t1 = std::chrono::high_resolution_clock::now();
  printf("Built AMGCL solve object and preconditioner in %f seconds\n", (float)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.);
  #endif
}

AMGCLSolver::~AMGCLSolver() {
  if (solver_impl)
    delete solver_impl;
  #ifdef USE_CUDA
  if (ctx)
    delete ctx;
  #endif
}

void AMGCLSolver::Solve(const Eigen::SparseMatrix<T, Eigen::RowMajor, nm>& A_eigen_reduced, Eigen::Matrix<T, Eigen::Dynamic, 1>& x, const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) {
  if (solver_impl)
    delete solver_impl;

  n = A_eigen_reduced.rows();

  // Use AMGCL's Eigen adapter to get the matrix into their internal format
  #ifdef USE_MPI
  solver_impl = new AMGCLSolverImpl(comm, A_eigen_reduced, prm, bprm);
  #else
  solver_impl = new AMGCLSolverImpl(A_eigen_reduced, prm, bprm);
  #endif
  #ifdef SOLVER_DEBUG
  printf("Made solve obj...\n");
  #endif

  Solve(x, b);
  
  printf("Building RHS...\n");
  Eigen::VectorXd resid = A_eigen_reduced*x;
  printf("initial residual / norm b is %f\n", resid.norm());


}

void AMGCLSolver::Solve(Eigen::Matrix<T, Eigen::Dynamic, 1>& x, const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) {
  // printf("Copying eigen data into vectors...\n");
  // std::vector<int> ptr, col;
  // std::vector<T> val;

   printf("A EIGEN REDUCED ROWS = %d\n", n);
  // printf("A EIGEN REDUCED COLS = %lu\n", A_eigen_reduced.cols());
   printf("xr = %lu\n", x.rows());
   printf("br = %lu\n", b.rows());

  // ptr.assign(A_eigen_reduced.outerIndexPtr(), A_eigen_reduced.outerIndexPtr() + n + 1);
  // ptr[n] = A_eigen_reduced.nonZeros();
  // col.assign(A_eigen_reduced.innerIndexPtr(), A_eigen_reduced.innerIndexPtr() + A_eigen_reduced.nonZeros());
  // val.assign(A_eigen_reduced.valuePtr(), A_eigen_reduced.valuePtr() + A_eigen_reduced.nonZeros());
  // printf("Done with that...\n");

  // printf("Building RHS...\n");
  // Eigen::VectorXd resid = A_eigen_reduced*x;
  // printf("initial residual / norm b is %f\n", resid.norm());

  // Eigen::VectorXd r = Eigen::VectorXd::Random(n);
  // Eigen::VectorXd _b = A_eigen_reduced*r;
  // printf("initial residual / norm _b is %f\n", _b.norm());
  // vex::vector<T> b_vexcl(n, _b.data());

  //for(int i=0;i<(int)n;++i)
    //b.coeffRef(i)*=10;
  Eigen::VectorXd _b = config.amgcl_rhs_scaling * b;

  printf("initial residual / norm _b is %f\n", _b.norm());

  // for(int i=0;i<(int)n;++i)
  //     std::cout<<_b[i] << "\t" << b[i] << std::endl;

  #ifdef USE_CUDA
  // Create device vector for b
  vex::vector<T> b_vexcl(n, _b.data());
  // std::vector<double> _b_v(_b.data(), _b.data() + _b.size());
  // auto b_cuda = Backend::copy_vector(_b_v, bprm);
  #ifdef SOLVER_DEBUG
  printf("Done with that...\n");
  #endif
  #endif

  // for(int i=0;i<n;++i)
  //    rhs[i] = b[i]; //b.data(), b.data() + n);
  // std::generate(rhs.begin(), rhs.end(), std::rand);
  // T mean = std::accumulate( rhs.begin(), rhs.end(), (T)0)/rhs.size();
  // for(auto& element : rhs)
  //    element -= mean;

  // std::cout<<"A_eigen_reduced: " << A_eigen_reduced.rows() << " " << A_eigen_reduced.cols() << std::endl;

  // vex::vector<T> x_vexcl(ctx, n); x *= T(0.0);
  Eigen::Matrix<T, Eigen::Dynamic, 1> xr = Eigen::MatrixXd::Random(n, 1);
  // std::cout<<"x before solve = "<<xr<<std::endl;
  #ifdef USE_CUDA
  vex::vector<T> x_vexcl(n, xr.data());
  // std::vector<double> _x_v(xr.data(), xr.data() + xr.size());
  // auto x_cuda = Backend::copy_vector(_x_v, bprm);
  #endif
  size_t iters;
  double error;

  #ifdef SOLVER_DEBUG
  auto t0 = std::chrono::high_resolution_clock::now();
  printf("copied vecs to device...\n");
  #endif
  #ifdef USE_CUDA
  std::tie(iters, error) = solver_impl->operator()(b_vexcl, x_vexcl);
  // std::tie(iters, error) = solver_impl->operator()(*b_cuda, *x_cuda);
  #else
  // auto f_rhs = Backend::copy_vector(b, bprm);
  // auto f_x = Backend::copy_vector(x, bprm);
  // std::tie(iters, error) = solver_impl->operator()(*f_rhs, *f_x);
  std::tie(iters, error) = solver_impl->operator()(_b, xr);
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
  // vex::copy(x_vexcl.begin(), x_vexcl.end(), x.begin());
  // std::vector<T> hxr(n);
  // vex::copy(x_vexcl, hxr);
  // std::cout<<"hxr = " ; for(int i=0;i<n;++i)std::cout<< hxr[i]<<" ";std::cout << std::endl;
}

void AMGCLSolver::setParams() {
  std::cout << "CFG TOL = " << config.tol << std::endl;
  std::cout << "CFG MAX ITER = " << config.max_iter << std::endl;

  #ifdef USE_CUDA
  bprm.q = *ctx;
  // cusparseCreate(&bprm.cusparse_handle);
  //{
  //    int dev = 0;//mpi_domain.world_rank % 6;
  //    //cudaGetDevice(&dev);
  //    cudaSetDevice(dev);

  //    cudaDeviceProp prop;
  //    cudaGetDeviceProperties(&prop, dev);
  //    std::cout << prop.name << std::endl << std::endl;
  //}
  #endif

  prm.put("solver.tol", config.tol);
  prm.put("solver.maxiter", config.max_iter);
  prm.put("precond.class", "amg");
  prm.put("precond.relax.type", "chebyshev");
  prm.put("precond.relax.degree", 2);
  prm.put("precond.relax.power_iters", 100);
  prm.put("precond.relax.higher", 2.0f);
  prm.put("precond.relax.lower", 1.0f / 120.0f);
  prm.put("precond.relax.scale", true);
  prm.put("precond.max_levels", 6);

  prm.put("precond.direct_coarse", false);
  // prm.put("solver.type", "cg");

  prm.put("precond.coarsening.type", "smoothed_aggregation");
  prm.put("precond.coarsening.estimate_spectral_radius", true);
  prm.put("precond.coarsening.relax", 1.0f);
  prm.put("precond.coarsening.power_iters", 100);
  // prm.put("precond.coarse_enough", 5000);
  prm.put("precond.ncycle", 2);

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



