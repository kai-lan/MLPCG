#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "AMGCLSolver.h"
#include "BinaryIO.h"

namespace py = pybind11;

using AMGCL = AMGCLSolver<PBackend, SBackend>;

PYBIND11_MODULE(pyamgcl_cuda, m) {
    m.doc() = "python binding for AMGCL solver CUDA version";
    m.def("solve", [](const SpMat& A, const VXT& b, double tol=1e-4, double atol=1e-10, int max_iters=100, bool verbose=false) {
        VXT x(b.size());
        x.setZero();
        auto info = AMGCL::Solve(A, x, b, tol, atol, max_iters, verbose);
        return std::make_tuple(x, info);
    });
}

