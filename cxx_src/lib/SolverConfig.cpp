#include "SolverConfig.h"

void SolverConfig::DefineOptions(cxxopts::Options& options) {
  // clang-format off
  options.add_options()
    ("solver", "Linear solver to use (cg, gmres, LU)", cxxopts::value<std::string>()->default_value("cg"))
    ("lib", "Library to use (trilinos, eigen)", cxxopts::value<std::string>()->default_value("eigen"))
    ("use_precond", "Set to use a preconditioner", cxxopts::value<bool>())
    ("diag_ones", "Put 1s in diagonal entries at intersection of zero rows+columns", cxxopts::value<bool>())
    ("remove_zrc", "Remove zero rows and columns", cxxopts::value<bool>())
    ("specify_colmap", "Specify col map for A", cxxopts::value<bool>())
    ("abs_tol", "Use absolute not relative tolerance for residual (|r| not |r|/|r_0|)", cxxopts::value<bool>())
    ("max_iter", "Max number of iterations for linear solve", cxxopts::value<int>()->default_value("250"))
    ("tol", "Tolerance for linear solve", cxxopts::value<T>()->default_value("0.000001"))
    ("no_pd_check", "Don't assert for positive definiteness during Belos CG solve", cxxopts::value<bool>())
    ("rpc", "Use right instead of left preconditioner", cxxopts::value<bool>())
    ("amgcl_rhs_scaling", "", cxxopts::value<T>()->default_value("1"))
    ("amgcl_rhs_autoscale", "", cxxopts::value<bool>())
    ("amgcl_cg_project", "", cxxopts::value<bool>())
    ("amgcl_precond_coarsening_type", "", cxxopts::value<std::string>()->default_value(""))
    ("amgcl_precond_relaxation_type", "", cxxopts::value<std::string>()->default_value(""))
    ("amgcl_precond_coarse_enough", "", cxxopts::value<T>()->default_value("-1"))
    ("amgcl_precond_max_levels", "", cxxopts::value<int>()->default_value("-1"))
    ("amgcl_precond_direct_coarse", "", cxxopts::value<bool>())
    ("amgcl_solver_direct_coarse", "", cxxopts::value<bool>())
    ("amgcl_precond_class", "", cxxopts::value<std::string>()->default_value(""))
  ;
  // clang-format on
}

void SolverConfig::ParseConfig(cxxopts::ParseResult opts) {
  // Parse
  use_precond = (bool)opts["use_precond"].count();
  diag_ones = (bool)opts["diag_ones"].count();
  remove_zrc = (bool)opts["remove_zrc"].count();
  solver = opts["solver"].as<std::string>();
  lib = opts["lib"].as<std::string>();
  max_iter = opts["max_iter"].as<int>();
  specify_colmap = (bool)opts["specify_colmap"].count();
  abs_tol = (bool)opts["abs_tol"].count();
  tol = opts["tol"].as<T>();
  no_pd_check = (bool)opts["no_pd_check"].count();
  rpc = (bool)opts["rpc"].count();
  amgcl_rhs_scaling = opts["amgcl_rhs_scaling"].as<T>();
  amgcl_rhs_autoscale = (bool)opts["amgcl_rhs_autoscale"].count();
  amgcl_cg_project = (bool)opts["amgcl_cg_project"].count();
  amgcl_precond_coarsening_type = opts["amgcl_precond_coarsening_type"].as<std::string>();
  amgcl_precond_relaxation_type = opts["amgcl_precond_relaxation_type"].as<std::string>();
  amgcl_precond_coarse_enough = opts["amgcl_precond_coarse_enough"].as<T>();
  amgcl_precond_max_levels = opts["amgcl_precond_max_levels"].as<int>();
  amgcl_precond_direct_coarse = (bool)opts["amgcl_precond_direct_coarse"].count();
  amgcl_solver_direct_coarse = (bool)opts["amgcl_solver_direct_coarse"].count();
  amgcl_precond_class = opts["amgcl_precond_class"].as<std::string>();
}

