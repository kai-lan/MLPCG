#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "AMGCLSolver.h"
#include "BinaryIO.h"
#include "MPIDomain.h"
#include "SolverConfig.h"

namespace py = pybind11;

using AMGCL = AMGCLSolver<PBackend, SBackend>;

PYBIND11_MODULE(pyamgcl_cuda, m) {
    m.doc() = "python binding for AMGCL solver CUDA version";
    py::class_<AMGCL>(m, "AMGCLSolverCUDA")
        .def(py::init<const SolverConfig&>())
        .def(py::init<const SpMat&, const SolverConfig&>())
        .def("print", [](AMGCL& solver, const SpMat& A) {
            std::cout << A << std::endl;
        })
        .def("solve", [](AMGCL& solver, const SpMat& A, const VXT& b)
            {
                VXT x(b.size());
                x.setZero();
                auto info = solver.Solve(A, x, b, false);
                return std::make_tuple(x, info);
            }, py::arg("A"), py::arg("b"));

}