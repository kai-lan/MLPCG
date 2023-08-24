#include "AMGCLSolver.h"
// #include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
// #include <amgcl/make_solver.hpp>
// #include <amgcl/amg.hpp>
// #include <amgcl/coarsening/smoothed_aggregation.hpp>
// #include <amgcl/relaxation/spai0.hpp>
// #include <amgcl/solver/bicgstab.hpp>

#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

#include "BinaryIO.h"
#include "MPIDomain.h"
#include "SolverConfig.h"
#include <unsupported/Eigen/SparseExtra>

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



int main(int argc, char* argv[]){
	typedef amgcl::make_solver<
    amgcl::amg<
      PBackend,
      amgcl::coarsening::smoothed_aggregation,
      amgcl::relaxation::spai0
      >,
    amgcl::solver::bicgstab<SBackend>
    > Solver;

	cxxopts::Options options("amgcl", "test");
   	SolverConfig config;
	config.max_iter = 100;
   	config.tol = 1e-4;

	VXT rhs;
	// IO::Eigen::Deserialize(rhs, config.rhs);
	Eigen::loadMarketVector(rhs, "../test_data/poisson3Db/b.mtx");

	// ptrdiff_t rows, cols;
    // std::vector<ptrdiff_t> ptr, col;
    // std::vector<double> val, rhs;
	amgcl::profiler<> prof("AMGCL solver");

	// prof.tic("read");
    // std::tie(rows, cols) = amgcl::io::mm_reader("../test_data/poisson3Db/A.mtx")(ptr, col, val);

    // std::tie(rows, cols) = amgcl::io::mm_reader("../test_data/poisson3Db/b.mtx")(rhs);
    // prof.toc("read");

	// auto A = std::tie(rows, ptr, col, val);

	SBackend::params bprm;
    Solver::params prm;
    prm.solver.tol = 1e-4;


	int dim = rhs.size();
	VXT x(dim);
	x.setZero();
    // std::vector<double> x(rows, 0.0);

	SpMat A(dim,dim);
   	// IO::Eigen::Deserialize(A, config.matrix);
	Eigen::loadMarket(A, "../test_data/poisson3Db/A.mtx");

	// prof.tic("setup");
    // Solver solve(A, prm, bprm);
    // prof.toc("setup");

	// int iters;
    // double error;
	// prof.tic("solve");
    // std::tie(iters, error) = solve(A, rhs, x);
    // prof.toc("solve");

	// std::cout << "Iters: " << iters << std::endl
    //           << "Error: " << error << std::endl
    //           << prof << std::endl;

	AMGCLSolver<PBackend, SBackend> amgcl(config);
	amgcl.Solve(A, x, rhs, true);
	// auto r = rhs - A * x;

	// for (int i = 0; i < x.size(); ++i) {
	// 	assert(std::isnan(x[i]));
	// }

	// std::cout << "Abs Residual: " << r.norm() << std::endl;
	// std::cout << "Rel Residual: " << r.norm() / rhs.norm() << std::endl;

	// int iters = 100;
	// auto t0 = std::chrono::high_resolution_clock::now();
	// for (int i = 0; i < iters; ++i)
	// 	amgcl.Solve(A, x, rhs);
	// auto t1 = std::chrono::high_resolution_clock::now();
	// std::cout << "Solving took " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()/1000.0 / iters << " s" << std::endl;

	// auto info = amgcl.Solve(A, x, rhs, true);
	// std::cout << std::get<0>(info) << ", " << std::get<1>(info) << std::endl;

	// auto r = rhs - A * x;
	// std::cout << "residual " << r.norm() << std::endl;
	return 0;
}
