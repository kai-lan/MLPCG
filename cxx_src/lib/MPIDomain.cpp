#include "MPIDomain.h"

#include <algorithm>
#include <climits>
#include <numeric>

//#include "VerbosityLogger.h"

//#define DEBUG_MPI

MPIDomain::MPIDomain() {}

MPIDomain::MPIDomain(const Index& _global_nodes_per_dir, T _dx) : global_nodes_per_dir(_global_nodes_per_dir), dx(_dx) {
  #ifdef USE_MPI
  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  #ifdef ONE_D
  #elif defined TWO_D
  #else
  #endif
  // ranks_per_axis = (int)std::pow(T(world_size), T(1)/T(d));

  ranks_per_axis = Index();

  SplitGrid();

  // Get the rank of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  printf("MPI initialized on host %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);
  #else
  world_size = 1;
  world_rank = 0;
  ranks_per_axis.fill(1);
  #endif
}

 MPIDomain::~MPIDomain() {}

void  MPIDomain::FillNeighbors() {
  for (int i = -1; i <= 1; ++i) {
    Index idx;
    #ifndef ONE_D
    for (int j = -1; j <= 1; ++j) {
      #ifndef TWO_D
      for (int k = -1; k <= 1; ++k) {
        idx = {i, j, k};
        #else
        idx = {i, j};
        #endif
        #else
        idx = {i};
        #endif

        bool valid = true;
        Index sum = rank_offset_per_dir;
        for (size_t dd = 0; dd < d; ++dd) {
          sum[dd] += idx[dd];
          if (sum[dd] < 0 || sum[dd] >= ranks_per_axis[dd])
            valid = false;
        }
        if (valid) {
          int fri = FlatRankIndex(sum);
          if (fri != world_rank)
            neighbor_ranks.push_back(fri);
        }

        #ifndef ONE_D
        #ifndef TWO_D
      }
      #endif
    }
    #endif
  }

  #ifdef DEBUG_MPI
  std::cout << "Neighbors for rank " << world_rank << ": ";
  for (size_t i = 0; i < neighbor_ranks.size(); ++i)
    std::cout << neighbor_ranks[i] << " ";
  std::cout << std::endl;
  #endif
}

int  MPIDomain::FlatRankIndex(const Index& idx) const {
  #ifdef ONE_D
  return idx[0];
  #elif defined TWO_D
  return idx[0] * ranks_per_axis[1] + idx[1];
  #else
  return idx[0] * ranks_per_axis[1] * ranks_per_axis[2] + idx[1] * ranks_per_axis[2] + idx[2];
  #endif
}

// Adapted from PhysBAM::MPI_GRID
void  MPIDomain::Minimize2DSurfaceArea(const int wsize, const int m, const int n, int& count_x) {
  count_x = std::max(1, (int)sqrt((T)wsize * m / n));
  if (wsize % count_x == 0) {
    if (wsize % (count_x + 1) == 0 && n * count_x + m * wsize / count_x >= n * (count_x + 1) + m * wsize / (count_x + 1))
      count_x++;
  } else if (wsize % (count_x + 1) == 0)
    count_x++;
  else
    count_x = m > n ? wsize : 1;
}

// Adapted from PhysBAM::MPI_GRID
void  MPIDomain::SplitGrid() {
  #ifdef ONE_D
  TGSLAssert(false, "MPIDomain: SplitGrid not implemented.");
  #elif defined TWO_D
  int m = global_nodes_per_dir[0], n = global_nodes_per_dir[1];
  Index zero = {0, 0};
  if (ranks_per_axis != zero)
    return;
  else {  // try to figure out counts by minimizing surface area between processes
    int xr = ranks_per_axis[0];
    Minimize2DSurfaceArea(world_size, m, n, xr);
    ranks_per_axis[0] = xr;
    ranks_per_axis[1] = world_size / xr;
  }
  TGSLAssert(ranks_per_axis[0] * ranks_per_axis[1] == world_size, "SplitGrid didn't partition domain properly");
  std::cout << "dividing domain into " << ranks_per_axis[0] << " by " << ranks_per_axis[1] << " processor grid" << std::endl;
  #else
  int m = global_nodes_per_dir[0], n = global_nodes_per_dir[1], mn = global_nodes_per_dir[2];
  Index count;
  Index zero = {0, 0, 0};
  if (ranks_per_axis != zero)
    return;
  else {  // try to figure out counts by minimizing surface area between processes
    int minimum_surface_area = INT_MAX;
    Index test_count = {0, 0, 0};
    for (test_count[2] = 1; test_count[2] <= world_size; test_count[2]++)
      if (world_size % test_count[2] == 0) {
        int tc0 = test_count[0];
        Minimize2DSurfaceArea(world_size / test_count[2], m, n, tc0);
        test_count[0] = tc0;
        test_count[1] = world_size / (test_count[0] * test_count[2]);
        int surface_area = test_count[0] * (n * mn) + test_count[1] * (m * mn) + test_count[2] * (m * n);
        if (surface_area < minimum_surface_area) {
          count = test_count;
          minimum_surface_area = surface_area;
        }
        int tc1 = test_count[1];
        Minimize2DSurfaceArea(world_size / test_count[2], n, m, tc1);
        test_count[1] = tc1;
        test_count[0] = world_size / (test_count[1] * test_count[2]);
        surface_area = test_count[0] * (n * mn) + test_count[1] * (m * mn) + test_count[2] * (m * n);
        if (surface_area < minimum_surface_area) {
          count = test_count;
          minimum_surface_area = surface_area;
        }
      }
    if (minimum_surface_area == INT_MAX) {
      std::cout << "Don't know how to divide domain in all directions." << std::endl;
      exit(1);
    }
  }
  TGSLAssert(count[0] * count[1] * count[2] == world_size, "SplitGrid didn't partition domain properly");
  std::cout << "dividing domain into " << count[0] << " by " << count[1] << " by " << count[2] << " processor grid" << std::endl;
  ranks_per_axis = count;
  #endif
}

void  MPIDomain::Initialize(Index& grid_N, const T& _dx) {
  global_nodes_per_dir = grid_N;
  dx = _dx;

  #ifdef ONE_D
  Index zero = {0};
  TGSLAssert(grid_N != zero, "MPIDomain::initialize was called with zero grid_N");
  #elif defined TWO_D
  Index zero = {0, 0};
  TGSLAssert(grid_N != zero, "MPIDomain::initialize was called with zero grid_N");
  #else
  Index zero = {0, 0, 0};
  TGSLAssert(grid_N != zero, "MPIDomain::initialize was called with zero grid_N");
  #endif

  // SplitGrid should have already been called at this point,
  // so we have global_nodes_per_dir and ranks_per_axis

  #ifdef USE_MPI
  std::cout << "global_nodes_per_dir = " << global_nodes_per_dir[0] << " " << global_nodes_per_dir[1] << " " << (d == 3 ? std::to_string(global_nodes_per_dir[2]) : "")
            << ", rpa = " << ranks_per_axis[0] << " " << ranks_per_axis[1] << " " << (d == 3 ? std::to_string(ranks_per_axis[2]) : "") << std::endl;
  for (size_t i = 0; i < d; ++i)
    N_local[i] = (nm)global_nodes_per_dir[i] / ranks_per_axis[i];  // TODO WARNING dabh this may be funky if it doesn't divide properly

  // TODO dabh reenable if rank 2 breaks w/o this
  // if (world_size == 2)
  //    N_local = (nm)global_nodes_per_dir*(nm)global_nodes_per_dir / 2;

  // grid_N.fill(N_local + ghost_width);
  // grid_N.fill(global_nodes_per_dir);

  #ifdef TWO_D
  // n ranks = 2 is special case
  if (world_size == 2) {
    if (world_rank == 0) {
      rank_offset_per_dir[0] = 0;
      rank_offset_per_dir[1] = 0;
    } else {
      rank_offset_per_dir[0] = 1;
      rank_offset_per_dir[1] = 0;
    }
  } else {
    // prefer walking in the y direction first
    rank_offset_per_dir[0] = world_rank / ranks_per_axis[1];
    rank_offset_per_dir[1] = world_rank % ranks_per_axis[1];
  }
  #else
  // prefer walking in the z direction first, then y
  rank_offset_per_dir[0] = world_rank / (ranks_per_axis[1] * ranks_per_axis[2]);
  rank_offset_per_dir[1] = (world_rank % (ranks_per_axis[1] * ranks_per_axis[2])) / ranks_per_axis[2];
  rank_offset_per_dir[2] = world_rank % ranks_per_axis[2];
  #endif

  // TODO this if branch may still only work for square domain
  if (world_size == 2) {
    if (world_rank == 0) {
      global_spatial_offset[0] = 0;
      global_spatial_offset[1] = 0;
      if (d == 3)
        global_spatial_offset[2] = 0;
    } else {
      global_spatial_offset[0] = T(0.5) * ((grid_N[0] - 1) * dx);
      global_spatial_offset[1] = 0;
      if (d == 3)
        global_spatial_offset[2] = 0;
    }

    local_min_corner = global_spatial_offset;
    if (world_rank == 0) {
      local_max_corner[0] = T(0.5) * ((grid_N[0] - 1) * dx);
      local_max_corner[1] = T(1.) * ((grid_N[1] - 1) * dx);
      if (d == 3)
        local_max_corner[2] = T(1.) * ((grid_N[2] - 1) * dx);
    } else {
      local_max_corner[0] = T(1.) * ((grid_N[0] - 1) * dx);
      local_max_corner[1] = T(1.) * ((grid_N[1] - 1) * dx);
      if (d == 3)
        local_max_corner[2] = T(1.) * ((grid_N[2] - 1) * dx);
    }

    if (world_rank == 0) {
      min_cell_index[0] = 0;
      min_cell_index[1] = 0;
      if (d == 3)
        min_cell_index[2] = 0;
      max_cell_index[0] = global_nodes_per_dir[0] / 2 - 1 - 1;  // global_nodes_per_dir-1;
      max_cell_index[1] = global_nodes_per_dir[1] - 1 - 1;      // global_nodes_per_dir/2 - 1;
      if (d == 3)
        max_cell_index[2] = global_nodes_per_dir[2] - 1 - 1;
    } else {
      min_cell_index[0] = global_nodes_per_dir[0] / 2 - 1;  // 0;
      min_cell_index[1] = 0;                                // global_nodes_per_dir/2;
      if (d == 3)
        min_cell_index[2] = 0;
      max_cell_index[0] = global_nodes_per_dir[0] - 1 - 1;  // global_nodes_per_dir-1;
      max_cell_index[1] = global_nodes_per_dir[1] - 1 - 1;  // global_nodes_per_dir-1;
      if (d == 3)
        max_cell_index[2] = global_nodes_per_dir[2] - 1 - 1;
    }
  } else {
    for (size_t i = 0; i < d; ++i) {
      // global_spatial_offset[i] = ((N_local[i]*dx)-(T(0.5)*dx)) * rank_offset_per_dir[i];
      global_spatial_offset[i] = rank_offset_per_dir[i] * (((T(global_nodes_per_dir[i] - 1)) / global_nodes_per_dir[i]) * N_local[i] *
                                                           dx);  //((T(N_local[i])/T(global_nodes_per_dir[i]))*rank_offset_per_dir[i])*((global_nodes_per_dir[i]-1)*dx);// (N_local[i])*dx *
                                                                 // rank_offset_per_dir[i] - T(N_local[i]*(rank_offset_per_dir[i]))/global_nodes_per_dir[i] * dx;
                                                                 // global_spatial_offset[i] = (N_local[i]*dx) * rank_offset_per_dir[i];// - T(0.5)*dx;
      // global_spatial_offset[i] = std::max(global_spatial_offset[i], T(0));
    }

    local_min_corner = global_spatial_offset;
    for (size_t i = 0; i < d; ++i) {
      #ifdef DEBUG_MPI
      std::cout << "rk " << world_rank << " local max corner " << i << " " << local_min_corner[i] << " " << N_local[i] << " " << dx << std::endl;
      #endif
      local_max_corner[i] =
        local_min_corner[i] +
        (((T(global_nodes_per_dir[i] - 1)) / global_nodes_per_dir[i]) * N_local[i] *
         dx);  // + ((T(N_local[i])/T(global_nodes_per_dir[i])))*((global_nodes_per_dir[i]-1)*dx); // + (N_local[i])*dx - (T(N_local[i]*(rank_offset_per_dir[i]))/global_nodes_per_dir[i]) * dx;
               // local_max_corner[i] = local_min_corner[i] + (N_local[i])*dx;// - (world_size > 1 ? (T(0.5)*dx) : 0);
      // local_max_corner[i] = std::min(local_max_corner[i], (T)1);
    }

    for (size_t i = 0; i < d; ++i) {
      min_cell_index[i] = rank_offset_per_dir[i] * N_local[i];
      max_cell_index[i] = std::min(min_cell_index[i] + N_local[i] - 1, grid_N[i] - 1 - 1);
    }
  }

  local_min_corner_with_ghost = local_min_corner;
  local_max_corner_with_ghost = local_max_corner;
  min_cell_index_with_ghost = min_cell_index;
  max_cell_index_with_ghost = max_cell_index;
  for (size_t i = 0; i < d; ++i) {
    if (local_min_corner[i] > 0 + T(1e-13)) {
      local_min_corner_with_ghost[i] -= ghost_width * dx;
      min_cell_index_with_ghost[i] -= ghost_width;
    }
    if (local_max_corner[i] < (global_nodes_per_dir[i] - 1) * dx - T(1e-13)) {
      local_max_corner_with_ghost[i] += ghost_width * dx;
      max_cell_index_with_ghost[i] += ghost_width;
    }
  }
  for (size_t i = 0; i < d; ++i) {
    min_cell_index_with_ghost[i] = std::max(min_cell_index_with_ghost[i], nm(0));
    max_cell_index_with_ghost[i] = std::min(max_cell_index_with_ghost[i], nm(global_nodes_per_dir[i] - 1 - 1));
  }

  p_per_rank.resize(world_size);
  lambda_per_rank.resize(world_size);
  dofs_per_rank.resize(world_size);

  #ifdef TWO_D
  if (world_size == 2)
    p_per_rank[world_rank] = global_nodes_per_dir[0] * global_nodes_per_dir[1] / 2;
  else
    p_per_rank[world_rank] = std::accumulate(N_local.begin(), N_local.end(), 1, std::multiplies<nm>());
  #else
  if (world_size == 2)
    p_per_rank[world_rank] = global_nodes_per_dir[0] * global_nodes_per_dir[1] * global_nodes_per_dir[2] / 2;
  else
    p_per_rank[world_rank] = std::accumulate(N_local.begin(), N_local.end(), 1, std::multiplies<nm>());
  #endif

  #ifdef TWO_D
  if (world_size == 2) {
    lambda_per_rank[world_rank] = global_nodes_per_dir[0] + global_nodes_per_dir[1] - 2;
  } else {
    if ((min_cell_index[0] == 0 && min_cell_index[1] == 0) || (max_cell_index[0] == grid_N[0] - 2 && min_cell_index[1] == 0) || (min_cell_index[0] == 0 && max_cell_index[1] == grid_N[1] - 2) ||
        (max_cell_index[0] == grid_N[0] - 2 && max_cell_index[1] == grid_N[1] - 2)) {
      if (min_cell_index[0] == 0 && min_cell_index[1] == 0 && max_cell_index[0] == grid_N[0] - 2 && max_cell_index[1] == grid_N[1] - 2)
        lambda_per_rank[world_rank] = 2 * (N_local[0] + N_local[1]) - 4;  // don't double count corner cell
      else
        lambda_per_rank[world_rank] = N_local[0] + N_local[1] - 1;  // don't double count corner cell
    } else if (min_cell_index[0] == 0 || max_cell_index[0] == grid_N[0] - 2) {
      lambda_per_rank[world_rank] = N_local[0];
    } else if (min_cell_index[0] == 0 || max_cell_index[0] == grid_N[0] - 2) {
      lambda_per_rank[world_rank] = N_local[1];
    } else {
      lambda_per_rank[world_rank] = 0;
    }
  }
  #else
  // Shouldn't need to fill out p and lambda per rank in here anymore because
  // they should get filled out by buildGlobalReducedMaps and buildBoundaryMaps

  /*if (world_size == 2){
      lambda_per_rank[world_rank] = global_nodes_per_dir[0] + global_nodes_per_dir[1] - 2;
  }else{
      int n_boundary_faces = 0;
      for (sz i = 0; i < d; ++i) {
          if (min_cell_index[i] == 0)
              n_boundary_faces++;
          if (max_cell_index[i] == grid_N[i] - 2)
              n_boundary_faces++;
      }

      if (n_boundary_faces == 0)
          lambda_per_rank[world_rank] = 0;
      else if (n_boundary_faces == 1)


      if (
          (min_cell_index[0] == 0 && min_cell_index[1] == 0 && min_cell_index[2] == 0) ||
          (max_cell_index[0] == grid_N[0]-2 && min_cell_index[1] == 0 && min_cell_index[2] == 0) ||
          (min_cell_index[0] == 0 && max_cell_index[1] == grid_N[1]-2) ||
          (max_cell_index[0] == grid_N[0]-2 && max_cell_index[1] == grid_N[1]-2)
      ) {
          if (min_cell_index[0] == 0 && min_cell_index[1] == 0 && max_cell_index[0] == grid_N[0]-2 && max_cell_index[1] == grid_N[1]-2)
              lambda_per_rank[world_rank] = 2*(N_local[0] + N_local[1]) - 4; // don't double count corner cell
          else
              lambda_per_rank[world_rank] = N_local[0] + N_local[1] - 1; // don't double count corner cell
      } else if (min_cell_index[0] == 0 || max_cell_index[0] == grid_N[0]-2) {
          lambda_per_rank[world_rank] = N_local[0];
      } else if (min_cell_index[0] == 0 || max_cell_index[0] == grid_N[0]-2) {
          lambda_per_rank[world_rank] = N_local[1];
      }else{
          lambda_per_rank[world_rank] = 0;
      }
  }*/
  #endif

  for (int i = 0; i < world_size; ++i) {
    MPI_Bcast(&(p_per_rank[i]), 1, MPI_UINT64_T, i, MPI_COMM_WORLD);
    MPI_Bcast(&(lambda_per_rank[i]), 1, MPI_UINT64_T, i, MPI_COMM_WORLD);
  }

  // Wait for all dofs_per_rank to synchronize
  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < world_size; ++i) {
    dofs_per_rank[i] = p_per_rank[i] + lambda_per_rank[i];
  }

  std::cout << "p, lambda, dofs per ranks synchronized!  dof results on rk " << world_rank << ": ";
  std::cout << p_per_rank[world_rank] << " " << lambda_per_rank[world_rank] << " ";
  for (int i = 0; i < world_size; ++i) {
    std::cout << dofs_per_rank[i] << " ";
  }
  std::cout << std::endl;
  #else
  N_local = grid_N;
  global_spatial_offset.fill(0);
  local_min_corner = global_spatial_offset;
  rank_offset_per_dir.fill(0);

  for (size_t i = 0; i < d; ++i) {
    local_max_corner[i] = local_min_corner[i] + (N_local[i] - 1) * dx;
  }

  for (size_t i = 0; i < d; ++i) {
    min_cell_index[i] = rank_offset_per_dir[i] * N_local[i];
    max_cell_index[i] = min_cell_index[i] + N_local[i] - 1 - 1;
  }

  local_min_corner_with_ghost = local_min_corner;
  local_max_corner_with_ghost = local_max_corner;
  min_cell_index_with_ghost = min_cell_index;
  max_cell_index_with_ghost = max_cell_index;
  for (size_t i = 0; i < d; ++i) {
    if (local_min_corner[i] > 0 + T(1e-13)) {
      local_min_corner_with_ghost[i] -= ghost_width * dx;
      min_cell_index_with_ghost[i] -= ghost_width;
    }
    if (local_max_corner[i] < (global_nodes_per_dir[i] - 1) * dx - T(1e-13)) {
      local_max_corner_with_ghost[i] += ghost_width * dx;
      max_cell_index_with_ghost[i] += ghost_width;
    }
  }
  for (size_t i = 0; i < d; ++i) {
    min_cell_index_with_ghost[i] = std::max(min_cell_index_with_ghost[i], nm(0));
    max_cell_index_with_ghost[i] = std::min(max_cell_index_with_ghost[i], nm(global_nodes_per_dir[i] - 1 - 1));
  }

  p_per_rank.resize(1);
  p_per_rank[0] = std::accumulate(global_nodes_per_dir.begin(), global_nodes_per_dir.end(), 1, std::multiplies<nm>());  // std::pow(global_nodes_per_dir, d);
  lambda_per_rank.resize(1);
  // TODO dabh no cut cell support yet
  #ifdef TWO_D
  lambda_per_rank[0] = 2 * global_nodes_per_dir[0] + 2 * global_nodes_per_dir[1] - 4;
  #else
  lambda_per_rank[0] = 4 * global_nodes_per_dir[0] + 4 * global_nodes_per_dir[1] + 4 * global_nodes_per_dir[2] +

                       2 * ((global_nodes_per_dir[0] - 1) * (global_nodes_per_dir[1] - 1)) + 2 * ((global_nodes_per_dir[0] - 1) * (global_nodes_per_dir[2] - 1)) +
                       2 * ((global_nodes_per_dir[1] - 1) * (global_nodes_per_dir[2] - 1))

                       - 8;
  #endif
  // std::pow(global_nodes_per_dir,d-2)*(4*std::pow(global_nodes_per_dir, 2) - 4);
  // if (d == 3)
  //    lambda_per_rank[0] += 2*std::pow(global_nodes_per_dir - 1, 2);
  dofs_per_rank.resize(1);
  dofs_per_rank[0] = p_per_rank[0] + lambda_per_rank[0];
  #endif

  FillNeighbors();

  #ifdef DEBUG_MPI
  #ifdef USE_MPI
  std::cout << "MPI enabled" << std::endl;
  #else
  std::cout << "MPI NOT ENABLED" << std::endl;
  #endif
  std::cout << "My spatial offset for rank " << world_rank << " is " << global_spatial_offset[0] << " " << global_spatial_offset[1] << " " << (d == 3 ? std::to_string(global_spatial_offset[2]) : "")
            << std::endl;
  std::cout << "My rpa = " << ranks_per_axis[0] << " " << ranks_per_axis[1] << " " << (d == 3 ? std::to_string(ranks_per_axis[2]) : "") << std::endl;
  std::cout << "My dx = " << dx << std::endl;
  std::cout << "N_local for rank " << world_rank << " is " << N_local[0] << " " << N_local[1] << " " << (d == 3 ? std::to_string(N_local[2]) : "") << std::endl;
  std::cout << "gw for rank " << world_rank << " is " << ghost_width << std::endl;
  std::cout << "local min corner for rank " << world_rank << " is " << local_min_corner[0] << " " << local_min_corner[1] << " " << (d == 3 ? std::to_string(local_min_corner[2]) : "") << std::endl;
  std::cout << "local max corner for rank " << world_rank << " is " << local_max_corner[0] << " " << local_max_corner[1] << " " << (d == 3 ? std::to_string(local_max_corner[2]) : "") << std::endl;
  std::cout << "gh local min corner for rank " << world_rank << " is " << local_min_corner_with_ghost[0] << " " << local_min_corner_with_ghost[1] << " "
            << (d == 3 ? std::to_string(local_min_corner_with_ghost[2]) : "") << std::endl;
  std::cout << "gh local max corner for rank " << world_rank << " is " << local_max_corner_with_ghost[0] << " " << local_max_corner_with_ghost[1] << " "
            << (d == 3 ? std::to_string(local_max_corner_with_ghost[2]) : "") << std::endl;
  std::cout << "min ci for rank " << world_rank << " is " << min_cell_index[0] << " " << min_cell_index[1] << " " << (d == 3 ? std::to_string(min_cell_index[2]) : "") << std::endl;
  std::cout << "min with ghost for rank " << world_rank << " is " << min_cell_index_with_ghost[0] << " " << min_cell_index_with_ghost[1] << " "
            << (d == 3 ? std::to_string(min_cell_index_with_ghost[2]) : "") << std::endl;
  std::cout << "max with ghost for rank " << world_rank << " is " << max_cell_index_with_ghost[0] << " " << max_cell_index_with_ghost[1] << " "
            << (d == 3 ? std::to_string(max_cell_index_with_ghost[2]) : "") << std::endl;
  std::cout << "max ci  for rank " << world_rank << " is " << max_cell_index[0] << " " << max_cell_index[1] << " " << (d == 3 ? std::to_string(max_cell_index[2]) : "") << std::endl;
  std::cout << "ppr0 = " << p_per_rank[0] << " lpr0 = " << lambda_per_rank[0] << " dpr0 = " << dofs_per_rank[0] << std::endl;
  #endif
}

void  MPIDomain::SetDOFCounts(const nm num_local_p_dofs, const nm num_local_lambda_dofs) {
  p_per_rank[world_rank] = num_local_p_dofs;
  lambda_per_rank[world_rank] = num_local_lambda_dofs;

  // Need to resynchronize DOF counts whenever we change them on any rank.
  // Assume that setDOFCounts is called by all ranks at the same time.
  #ifdef USE_MPI
  for (int i = 0; i < world_size; ++i) {
    MPI_Bcast(&(p_per_rank[i]), 1, MPI_UINT64_T, i, MPI_COMM_WORLD);
    MPI_Bcast(&(lambda_per_rank[i]), 1, MPI_UINT64_T, i, MPI_COMM_WORLD);
  }

  // Wait for all dofs_per_rank to synchronize
  MPI_Barrier(MPI_COMM_WORLD);
  #endif

  for (int i = 0; i < world_size; ++i) {
    dofs_per_rank[i] = p_per_rank[i] + lambda_per_rank[i];
  }

  #ifdef DEBUG_MPI
  std::cout << "SETTING DOF COUNTS FOR RANK " << world_rank << " " << num_local_p_dofs << " " << num_local_lambda_dofs << " dofo " << dofOffset() << std::endl;
  #endif

  #ifdef DEBUG_MPI
  std::cout << "p, lambda, dofs per ranks synchronized!  dof results on rk " << world_rank << ": ";
  std::cout << p_per_rank[world_rank] << " " << lambda_per_rank[world_rank] << " ";
  for (int i = 0; i < world_size; ++i) {
    std::cout << dofs_per_rank[i] << " ";
  }
  std::cout << std::endl;
  #endif
}

 nm  MPIDomain::DOFOffset() const {
  return DOFOffset(world_rank);
}

 nm  MPIDomain::DOFOffset(size_t rank) const {
  nm dofs_per_rank_i = 0;
  for (size_t i = 0; i < rank; ++i)
    dofs_per_rank_i += dofs_per_rank[i];
  return dofs_per_rank_i;
}

bool  MPIDomain::IsLocalWithGhost(const Index& idx) const {
  for (size_t i = 0; i < d; ++i)
    if (idx[i] < min_cell_index_with_ghost[i] || idx[i] > max_cell_index_with_ghost[i])
      return false;
  return true;
}

bool  MPIDomain::IsLocalPWithGhost(const Index& idx) const {
  for (size_t i = 0; i < d; ++i)
    if (idx[i] < min_cell_index_with_ghost[i] || idx[i] > max_cell_index_with_ghost[i] + 1)
      return false;
  return true;
}

bool  MPIDomain::IsLocalPWithGhost(const Particle& loc) const {
  for (size_t i = 0; i < d; ++i)
    if (loc[i] < local_min_corner_with_ghost[i] || loc[i] > local_max_corner_with_ghost[i])
      return false;
  return true;
}

bool  MPIDomain::IsLocal(const Index& idx) const {
  for (size_t i = 0; i < d; ++i)
    if (idx[i] < min_cell_index[i] || idx[i] > max_cell_index[i])
      return false;
  return true;
}

bool  MPIDomain::IsLocalP(const Index& idx) const {
  for (size_t i = 0; i < d; ++i)
    if (idx[i] < min_cell_index[i] || idx[i] > max_cell_index[i] + 1)
      return false;
  return true;
}

bool  MPIDomain::IsLocalP(const Particle& loc) const {
  for (size_t i = 0; i < d; ++i)
    if (loc[i] < local_min_corner[i] || loc[i] > local_max_corner[i])
      return false;
  return true;
}

bool  MPIDomain::IsLocalV(const Particle& loc) const {
  bool local = true;

  for (size_t i = 0; i < d; ++i)
    if (loc[i] < local_min_corner[i] || loc[i] > local_max_corner[i]) {
      if (!(min_cell_index[i] == 0 && std::abs(loc[i]) <= dx) && !(max_cell_index[i] == global_nodes_per_dir[i] - 2 && std::abs(loc[i] - (local_max_corner[i] + dx)) <= dx))
        local = false;
    }
  return local;
}

