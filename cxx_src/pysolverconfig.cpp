#include <pybind11/pybind11.h>
#include "SolverConfig.h"

namespace py = pybind11;

PYBIND11_MODULE(pysolverconfig, m) {
    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("tol", &SolverConfig::tol)
        .def_readwrite("max_iter", &SolverConfig::max_iter);
}
