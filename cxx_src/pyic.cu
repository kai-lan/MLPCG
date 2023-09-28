#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "ICSolver.h"
#include "BinaryIO.h"

namespace py = pybind11;

using IC = ICSolver<PBackend, SBackend>;

PYBIND11_MODULE(pyic_cuda, m) {
    m.doc() = "python binding for Incomplete Cholesky precondtioned solver CUDA version";
    m.def("solve", [](const SpMat& A, const VXT& b, double tol=1e-4, double atol=1e-10, int max_iters=100, bool verbose=false) {
        VXT x(b.size());
        x.setZero();
        auto info = IC::Solve(A, x, b, tol, atol, max_iters, verbose);
        return std::make_tuple(x, info);
    });
}

