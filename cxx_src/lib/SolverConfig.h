#ifndef SOLVER_CONFIG_H
#define SOLVER_CONFIG_H

#include <string>

#include <cxxopts.hpp>

#include <Definitions.h>


struct SolverConfig {
  bool use_precond;
  bool diag_ones;
  bool remove_zrc;
  std::string solver;
  std::string lib;
  int max_iter;
  bool specify_colmap;
  T abs_tol;
  T tol;
  bool no_pd_check;
  bool rpc;
  T amgcl_rhs_scaling;
  bool amgcl_rhs_autoscale;
  bool amgcl_cg_project;
  std::string amgcl_precond_coarsening_type;
  std::string amgcl_precond_relaxation_type;
  T amgcl_precond_coarse_enough;
  int amgcl_precond_max_levels;
  bool amgcl_precond_direct_coarse;
  bool amgcl_solver_direct_coarse;
  std::string amgcl_precond_class;
  bool eigen_recompute = false;

  std::string matrix;
  std::string rhs;

  virtual void DefineOptions(cxxopts::Options& options);
  virtual void ParseConfig(cxxopts::ParseResult opts);
};


#endif

