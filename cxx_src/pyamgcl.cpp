#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "AMGCLSolver.h"
#include "BinaryIO.h"
#include "MPIDomain.h"
#include "SolverConfig.h"

namespace py = pybind11;

PYBIND11_MODULE(pyamgcl_ext, m) {
    m.doc() = "python binding for AMGCL solver";
    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("tol", &SolverConfig::tol)
        .def_readwrite("max_iter", &SolverConfig::max_iter);
    py::class_<AMGCLSolver>(m, "AMGCLSolver")
        .def(py::init<const SolverConfig&>())
        .def(py::init<const SpMat&, const SolverConfig&>())
        .def("print", [](AMGCLSolver& solver, const SpMat& A) {
            std::cout << A << std::endl;
        })
        .def("solve", [](AMGCLSolver& solver, const SpMat& A, const VXT& b)
            {
                VXT x(b.size());
                x.setZero();
                auto info = solver.Solve(A, x, b, false);
                return std::make_tuple(x, info);
            }, py::arg("A"), py::arg("b"));
        // .def_property_readonly("n", &AMGCLSolver::n);
    // py::class_<


}