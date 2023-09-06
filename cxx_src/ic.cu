#include "ICSolver.h"
#include "BinaryIO.h"
#include <unsupported/Eigen/SparseExtra>

typedef ICSolver<PBackend, SBackend>::Solver Solver;

int main(int argc, char* argv[]){

	int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    std::cout << prop.name << std::endl;

	VXT rhs;
	// Eigen::loadMarketVector(rhs, argv[2]);
	IO::Eigen::Deserialize(rhs, argv[2]);
	SpMat A(rhs.size(),rhs.size());
	// Eigen::loadMarket(A, argv[1]);
	IO::Eigen::Deserialize(A, argv[1]);


	int dim = rhs.size();
	VXT x(dim);
	x.setZero();

	auto info = ICSolver<PBackend, SBackend>::Solve(A, x, rhs, 1e-4);

	auto r = rhs - A * x;
	std::cout << "Iterations " << std::get<0>(info) << std::endl;
	std::cout << "Abs residual " << r.norm() << std::endl;
	std::cout << "Rel residual " << r.norm() / rhs.norm() << std::endl;

	return 0;
}
