#include "AMGCLSolver.h"
#include "BinaryIO.h"
#include "MPIDomain.h"
#include "SolverConfig.h"

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

	cxxopts::Options options("amgcl", "test");
   	SolverConfig config;

	try {
		config.DefineOptions(options);
		auto result = options.parse(argc, argv);
		if (result.count("help")) {
		std::cout << options.help({""}) << std::endl;
		exit(0);
		}
		config.ParseConfig(result);
	} catch (const cxxopts::OptionException& e) {
		std::cout << "Error parsing options: " << e.what() << std::endl;
		std::cout << options.help({""}) << std::endl;
		exit(-1);
	}

	config.max_iter = 100;
   	config.tol = 1e-4;
	if (config.matrix == "")
		config.matrix = "../test_data/A_comp_999.bin";
	if (config.rhs == "")
		config.rhs = "../test_data/b_comp_999.bin";


	VXT rhs;
	IO::Eigen::Deserialize(rhs, config.rhs);

	int dim = rhs.size();
	VXT x(dim);
	x.setZero();

	SpMat A(dim,dim);
   	IO::Eigen::Deserialize(A, config.matrix);

	AMGCLSolver<PBackend, SBackend> amgcl(config);
	// amgcl.Solve(A, x, rhs);
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

	auto info = amgcl.Solve(A, x, rhs, true);
	std::cout << std::get<0>(info) << ", " << std::get<1>(info) << std::endl;

	auto r = rhs - A * x;
	std::cout << "residual " << r.norm() << std::endl;
	return 0;
}
