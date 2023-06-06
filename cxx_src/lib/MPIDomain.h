#ifndef MPI_DOMAIN_H
#define MPI_DOMAIN_H

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <Definitions.h>


class MPIDomain {
 public:
  MPIDomain();
  MPIDomain(const Index& global_nodes_per_dir, T dx);
  ~MPIDomain();

  void Initialize(Index& grid_N, const T& _dx);
  void SetDOFCounts(const nm num_local_p_dofs, const nm num_local_lambda_dofs);
  nm DOFOffset() const;
  nm DOFOffset(size_t rank) const;
  bool IsLocalWithGhost(const Index& idx) const;
  bool IsLocalPWithGhost(const Index& idx) const;
  bool IsLocalPWithGhost(const Particle& loc) const;
  bool IsLocal(const Index& idx) const;
  bool IsLocalP(const Index& idx) const;
  bool IsLocalP(const Particle& loc) const;
  bool IsLocalV(const Particle& loc) const;
  int FlatRankIndex(const Index& idx) const;

 private:
  void Minimize2DSurfaceArea(const int wsize, const int m, const int n, int& count_x);
  void SplitGrid();
  void FillNeighbors();

 public:
  int world_size;  // MPI world size, set by MPI not user
  int world_rank;  // MPI world rank, set by MPI not user
  Index ranks_per_axis;
  std::vector<int> neighbor_ranks;
  int ghost_width = 3;  // in terms of number of cells/nodes, not physical units
  sz total_local_reduced_2_global_count = 0;

  Index global_nodes_per_dir;
  T dx;

  Index N_local;

  Index rank_offset_per_dir;
  Vector global_spatial_offset;

  // Spatial corners of MPI partition
  Vector local_min_corner;
  Vector local_max_corner;
  Vector local_min_corner_with_ghost;
  Vector local_max_corner_with_ghost;

  Index min_cell_index;
  Index max_cell_index;
  Index min_cell_index_with_ghost;
  Index max_cell_index_with_ghost;

  std::vector<nm> dofs_per_rank;
  std::vector<nm> p_per_rank;
  std::vector<nm> lambda_per_rank;
};


#endif

