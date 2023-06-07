#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include <Definitions.h>
#include <core/algebra/MatricesAndVectors.h>
namespace IO {

inline void WriteRHS(const TVV& values, const std::string filename) {
  std::ofstream outstream (filename);
  if (outstream.is_open())
  {
      for (sz i=0; i<values.size();++i) {
        for (sz j=0; j<values[i].size(); ++j)
          outstream << values[i][j] << "\n";
      }
      outstream.close();
  }
  else throw std::runtime_error("Unable to open file " + filename);
}

inline void WriteMTX(const IVI2& indices, const TV& values, const sz rows, const sz cols, const sz nnz, const std::string filename) {
  std::ofstream outstream (filename);
  if (outstream.is_open())
  {
    outstream << rows << " " << cols << " " << nnz << "\n";
    for (sz i=0; i<indices.size();++i)
        outstream << indices[i][0]+1 << " " << indices[i][1]+1 << " " << values[i] << "\n";
    outstream.close();
  }
  else throw std::runtime_error("Unable to open file " + filename);
}

inline void WriteCSV(const TV& x, const TV& y, const std::string filename) {
  std::ofstream outstream(filename);
  if (outstream.good()) {
    size_t N = x.size();
    TGSLAssert(N <= y.size(), "mismatched array sizes for write CSV");
    outstream << "x,y\n";
    for (size_t i = 0; i < N; ++i) {
      outstream << x[i] << "," << y[i] << std::endl;
    }
    outstream.close();
  }
}

inline void WriteOBJ(const TVP& positions, const IV& mesh, const std::string& filename) {
  sz N = positions.size();
  std::ofstream outstream(filename);
  if (outstream.good()) {
    outstream << "# " << N << " points\n";
    for (sz i = 0; i < N; ++i) {
      outstream << "v";
      sz j;
      for (j = 0; j < positions[i].size(); ++j)
        outstream << " " << positions[i][j];
      for (; j < 3; ++j)
        outstream << " 0";
      outstream << std::endl;
    }
    for (sz e = 0; e < mesh.size() / 3; ++e) {
      outstream << "f " << mesh[3 * e] + 1 << " " << mesh[3 * e + 1] + 1 << " " << mesh[3 * e + 2] + 1;
      outstream << std::endl;
    }
    outstream.close();
  }
}

// write quadrilateral surface mesh in .obj file format (3D only)
inline void WriteOBJ(const TVP3& positions, const IVI4& mesh, const std::string& filename) {
  sz N = positions.size();
  std::ofstream outstream(filename);
  if (outstream.good()) {
    outstream << "# " << N << " points\n";
    for (sz i = 0; i < N; ++i) {
      outstream << "v" << " "
                << positions[i][0] << " "
                << positions[i][1] << " "
                << positions[i][2] << "\n";
    }
    for (sz e = 0; e < mesh.size(); ++e) {
      outstream << "f " << mesh[e][0] + 1 << " "
                        << mesh[e][1] + 1 << " "
                        << mesh[e][2] + 1 << " "
                        << mesh[e][3] + 1 << "\n";
    }
    outstream.close();
  }
}

template <typename VectorType, typename F1>
void WriteOBJ(const std::vector<VectorType>& positions, const std::vector<VectorType>& normals, const size_t N, const std::string& filename, F1 filter) {
  std::ofstream outstream(filename);
  if (outstream.good()) {
    outstream << "# " << N << " points\n"
              << "g\n";
    for (size_t i = 0; i < N; ++i) {
      if (filter(i)) {
        outstream << "v";
        size_t j;
        for (j = 0; j < positions[i].size(); ++j)
          outstream << " " << positions[i][j];
        for (; j < 3; ++j)
          outstream << " 0";
        outstream << std::endl;
      }
    }
    if (normals.size()) {
      for (size_t i = 0; i < N; ++i) {
        if (filter(i)) {
          outstream << "vn";
          size_t j;
          for (j = 0; j < normals[i].size(); ++j)
            outstream << " " << normals[i][j];
          for (; j < 3; ++j)
            outstream << " 0";
          outstream << std::endl;
        }
      }
    }
    outstream.close();
  }
}

template <typename VectorType>
void ReadOBJMesh(std::vector<VectorType>& positions, IV& mesh, size_t& N, const std::string& filename) {
  positions.resize(0);
  mesh.resize(0);
  std::ifstream instream;
  std::string line;
  char ch;
  instream.open(filename);
  if (!instream) {
    std::cerr << "Unable to open file " << filename << std::endl;
    exit(1);  // call system to stop
  }

  while (std::getline(instream, line)) {
    std::stringstream ss(line);
    if (!(ss >> ch))
      continue;
    if (ch != '#'){
      if (ch == 'v') {
        std::stringstream ss2(line);
        char ch2;
        ss2 >> ch2;
        ss2 >> ch2;
        if (ch2 == 't' || ch2 == 'n') {
          continue;
        }
        else {
          VectorType pos_p;
          for (sz j = 0; j < d; j++)
            ss >> pos_p[j];
          positions.emplace_back(pos_p);
        }
      }
      if (ch == 'f') {
        std::string next_token, v;
        for (sz j = 0; j < d; j++) {
          ss >> next_token;
          std::stringstream token_stream(next_token);
          std::getline(token_stream, v, '/');
          mesh.emplace_back(std::stoi(v) - 1); // .obj's use 1-based indexing for vertices
        }
      }
    }
  }
  N = positions.size();
}
template <typename VectorType>
void ReadTetehedralNodesTetgen(std::vector<VectorType>& positions, const std::string& filename, bool zero_based_index = false) {
  //Tetgen node file loader
  positions.resize(0);
  std::ifstream instream;
  std::string line;
  char ch;
  instream.open(filename);
  if (!instream) {
    std::cerr << "Unable to open file " << filename << std::endl;
    exit(1);  // call system to stop
  }

  std::getline(instream, line);
  std::stringstream s1(line);
  nm num_elements;
  s1 >> num_elements;
  positions.resize(num_elements);
  for(sz i =0; i<num_elements; i++){
    std::getline(instream, line);
    std::stringstream ss(line);
    sz element_number;
    ss >> element_number;
    element_number -= (1 - static_cast<int>(zero_based_index));
    TGSLAssert(i==element_number,"IO::ReadTetehedralNodesTetgen, ID and the element id number do not match");
    for(sz j =0; j<d; j++){
      T node;
      ss >> node;
      positions[i][j] = node;
    }
  }
  instream.close();
}

inline void ReadTetehedralMeshTetgen(IV& mesh, const std::string& filename, bool zero_based_index = false) {
  //Tetgen ele file loader
  std::ifstream instream;
  std::string line;
  //char ch;
  instream.open(filename);
  if (!instream) {
    std::cerr << "Unable to open file " << filename << std::endl;
    exit(1);  // call system to stop
  }

  std::getline(instream, line);
  std::stringstream s1(line);
  nm num_elements;
  s1 >> num_elements;
  mesh.resize(4*num_elements);
  for(sz i =0; i<sz(num_elements); i++){
    std::getline(instream, line);
    std::stringstream ss(line);
    sz element_number;
    ss >> element_number;
    element_number -= (1 - static_cast<int>(zero_based_index));
    TGSLAssert(i==element_number,"IO::ReadTetehedralMeshTetgen, ID and the element id number do not match");
    for(sz j =0; j<4; j++){
      sz node_id;
      ss >> node_id;
      mesh[4*i+j]= static_cast<TGSL::nm>(node_id - (1 - static_cast<int>(zero_based_index)));
    }
  }
  instream.close();
}

template <typename VectorType>
void ReadVTKTetewild(std::vector<VectorType>& positions, IV& mesh,const std::string& filename) {
  //Tetgen node file loader
  positions.resize(0);
  mesh.resize(0);
  std::ifstream instream;
  std::string line;
  char ch;
  instream.open(filename);
  if (!instream) {
    std::cerr << "Unable to open file " << filename << std::endl;
    exit(1);  // call system to stop
  }
  //File format
  //# vtk DataFile Version 2.0
  //metahuman_torso_, Created by Gmsh
  //ASCII
  //DATASET UNSTRUCTURED_GRID
  //POINTS 4914 double
  //
  std::getline(instream, line);
  std::getline(instream, line);
  std::getline(instream, line);
  std::getline(instream, line);
  std::getline(instream, line);
  std::stringstream s1(line);
  nm num_elements;
  std::string num_ss;
  s1 >> num_ss;
  //std::cout<< num_ss<<std::endl;
  s1 >> num_elements;
  //std::cout<< num_elements<<std::endl;
  positions.resize(num_elements);
  for(sz i =0; i<num_elements; i++){
    std::getline(instream, line);
    std::stringstream ss(line);
    for(sz j =0; j<d; j++){
      T node;
      ss >> node;
      positions[i][j] = node;
    }
  }
  std::getline(instream, line);
  std::getline(instream, line);
  std::stringstream s2(line);
  std::string cells;
  s2 >> cells;
  //std::cout<< cells<<std::endl;
  nm num_elements_mesh;
  s2 >> num_elements_mesh;
  //std::cout<< num_elements_mesh<<std::endl;
  mesh.resize(4*num_elements_mesh);
  for(sz i =0; i<sz(num_elements_mesh); i++){
  std::getline(instream, line);
  std::stringstream ss(line);
  sz element_number;
  ss >> element_number;
  //TGSLAssert(i==element_number,"IO::ReadTetehedralMeshTetgen, ID and the element id number do not match");
    for(sz j =0; j<4; j++){
      sz node_id;
      ss >> node_id;
      mesh[4*i+j]= node_id;
    }
  }
  instream.close();

}

inline void CleanOBJTriMesh(const std::string& filename) {
  // This function discards unnecessay texture/normal information and turns quads into triangles
  // Creates a new OBJ mesh (filename: new_XXX.obj)
  TVP x;
  IV mesh;
  std::ifstream instream;
  std::string line;
  char ch;
  instream.open(filename);
  if (!instream) {
    std::cerr << "Unable to open file " << filename << std::endl;
    exit(1);  // call system to stop
  }

  while (std::getline(instream, line)) {
    std::stringstream ss(line);
    ss >> ch;
    if (ch != '#'){
      if (ch == 'v') {
        Particle pos_p;
        for (sz j = 0; j < d; j++)
          ss >> pos_p[j];
        x.emplace_back(pos_p);
      }
      if (ch == 'f') {
        IV v_vector;
        std::string word;
        while (ss >> word) {
          size_t pos = 0;
          pos = word.find("/");
          v_vector.emplace_back(std::stoi(word.substr(0, pos)));
        }
        if (v_vector.size() == 3) {
          for (sz i = 0; i < 3; ++i)
            mesh.emplace_back(v_vector[i]-1);
        }
        if (v_vector.size() > 3) {
          for (sz i: {0,1,3})
            mesh.emplace_back(v_vector[i]-1);
          for (sz i: {0,2,3})
            mesh.emplace_back(v_vector[i]-1);
          for (sz i: {1,2,3})
            mesh.emplace_back(v_vector[i]-1);
        }
      }
    }
  }
  size_t pos = filename.find(".obj");
  WriteOBJ(x, mesh, filename.substr(0, pos) + "_new.obj");
}

template <typename VectorType>
void WriteOBJ(const std::vector<VectorType>& positions, const size_t N, const std::string& filename) {
  std::vector<VectorType> normals(0);
  IO::WriteOBJ(positions, normals, N, filename, [](const size_t i) { return true; });
}

// This version of IO::WriteOBJ writes a collection of line segments or rays; no faces/normals/etc.
inline void WriteOBJ(const std::vector<std::pair<Vector, Vector>>& rays, const std::string& filename){
  std::ofstream outstream(filename);
  if (outstream.good()) {
    outstream << "# " << 2*rays.size() << " points\n"
              << "g\n";
    for (size_t i = 0; i < rays.size(); ++i) {
      // Origin
      outstream << "v";
      size_t j;
      for (j = 0; j < rays[i].first.size(); ++j)
        outstream << " " << rays[i].first[j];
      for (; j < 3; ++j)
        outstream << " 0";
      outstream << std::endl;

      // Tip
      outstream << "v";
      for (j = 0; j < rays[i].second.size(); ++j)
        outstream << " " << rays[i].second[j];
      for (; j < 3; ++j)
        outstream << " 0";
      outstream << std::endl;
    }
    for (size_t i = 0; i < rays.size(); ++i) {
      outstream << "l " << (2*i+1) << " " << (2*i+1+1) << std::endl; // .obj files 1-index their vertices
    }
    outstream.close();
  }
}

inline void WriteOBJ(const std::vector<std::vector<std::pair<Vector, Vector>>>& rays, const std::string& filename){
  std::vector<std::pair<Vector, Vector>> flat;
  for (sz r = 0; r < rays.size(); ++r) {
    flat.insert(flat.end(), rays[r].begin(), rays[r].end());
  }
  WriteOBJ(flat, filename);
}

template <typename VectorType>
void WriteOBJ(const std::vector<VectorType>& positions, const std::vector<VectorType>& normals, const size_t N, const std::string& filename) {
  WriteOBJ(positions, normals, N, filename, [](const size_t i) { return true; });
}

template <typename VectorType>
void WritePLY(const std::vector<VectorType>& positions, const TVV& velocities, const TV& curlv, const TV& curvature, const TV& phi, const size_t N, const std::string& filename) {
  std::ofstream outstream(filename);
  if (outstream.good()) {
    outstream << "ply" << std::endl;
    outstream << "format ascii 1.0" << std::endl;
    outstream << "element vertex " << N << std::endl;
    outstream << "property double x" << std::endl;
    outstream << "property double y" << std::endl;
    outstream << "property double z" << std::endl;
    outstream << "property double vx" << std::endl;
    outstream << "property double vy" << std::endl;
    outstream << "property double vz" << std::endl;
    outstream << "property double curlv" << std::endl;
    outstream << "property double curvature" << std::endl;
    outstream << "property double phi" << std::endl;
    outstream << "end_header" << std::endl;

    for (size_t i = 0; i < N; ++i) {
      size_t j;
      for (j = 0; j < d; ++j)
        outstream << positions[i][j] << " ";
      for (; j < 3; ++j)
        outstream << "0 ";
      for (j = 0; j < d; ++j)
        outstream << velocities[i][j] << " ";
      for (; j < 3; ++j)
        outstream << "0 ";
      outstream << curlv[i] << " " << curvature[i] << " " << phi[i] << " ";
      outstream << std::endl;
    }
    outstream.close();
  }
}

template <typename VectorType>
void ReadPLY(std::vector<VectorType>& positions, TVV& velocities, TV& curlv, TV& curvature, TV& phi_p, size_t& N, const std::string& filename) {
  std::ifstream instream;
  std::string line;
  instream.open(filename);
  if (!instream) {
    std::cerr << "Unable to open file " << filename << std::endl;
    exit(1);  // call system to stop
  }
  std::getline(instream, line);
  std::getline(instream, line);
  std::getline(instream, line);
  std::stringstream ss(line);
  char ch;
  do {
    ss >> ch;
  } while (ch != 'x');
  ss >> N;
  do {
    std::getline(instream, line);
  } while (line != "end_header");
  positions.resize(N);
  velocities.resize(N);
  curlv.resize(N);
  curvature.resize(N);
  phi_p.resize(N);
  sz p = 0;
  sz j;
  T trash;
  while (!instream.eof()) {
    std::getline(instream, line);
    std::stringstream ss(line);
    for (j = 0; j < d; j++) {
      ss >> positions[p][j];
    }
    for (; j < 3; j++) {
      ss >> trash;
    }
    for (j = 0; j < d; j++) {
      ss >> velocities[p][j];
    }
    for (; j < 3; j++) {
      ss >> trash;
    }
    ss >> curlv[p];
    ss >> curvature[p];
    ss >> phi_p[p];
    for (; j < 3; j++) {
      ss >> trash;
    }
    p++;
  }
}

template <typename VectorType>
void WritePLY(const std::vector<VectorType>& positions, const TVV& velocity, const TV& curlv, const TV& radii, const std::vector<unsigned char>& use_B, const size_t N, const std::string& filename) {
  std::ofstream outstream(filename);
  if (outstream.good()) {
    outstream << "ply" << std::endl;
    outstream << "format ascii 1.0" << std::endl;
    outstream << "element vertex " << N << std::endl;
    outstream << "property double x" << std::endl;
    outstream << "property double y" << std::endl;
    outstream << "property double z" << std::endl;
    outstream << "property double vx" << std::endl;
    outstream << "property double vy" << std::endl;
    outstream << "property double vz" << std::endl;
    outstream << "property double curlv_x" << std::endl;
    outstream << "property double curlv_y" << std::endl;
    outstream << "property double curlv_z" << std::endl;
    outstream << "property double radii" << std::endl;
    outstream << "property bool use_B" << std::endl;
    outstream << "end_header" << std::endl;

    for (size_t i = 0; i < N; ++i) {
      size_t j;
      for (j = 0; j < positions[i].size(); ++j)
        outstream << positions[i][j] << " ";
      for (; j < 3; ++j)
        outstream << "0 ";
      for (j = 0; j < positions[i].size(); ++j)
        outstream << velocity[i][j] << " ";
      for (; j < 3; ++j)
        outstream << "0 ";
      #ifdef TWO_D
      outstream << curlv[i] << " ";
      #else
      for (j = 0; j < positions[i].size(); ++j)
        outstream << curlv[i * d + j] << " ";
      for (; j < 3; ++j)
        outstream << "0 ";
      #endif
      outstream << radii[i] << " " << use_B[i] << std::endl;
    }
    outstream.close();
  }
}

inline void WriteGridData(const TV& grid_data, const Index& grid_N, const T& dx, const std::string& filename, const size_t n_transfer = d) {
  std::ofstream outstream(filename);
  if (outstream.good()) {
    outstream << grid_N[0] << " " << grid_N[1];
    size_t num_grid = grid_N[0] * grid_N[1];
    #ifndef TWO_D
    num_grid *= grid_N[2];
    outstream << " " << grid_N[2];
    #endif
    outstream << " " << dx << std::endl;
    for (size_t i = 0; i < num_grid; ++i) {
      for (size_t j = 0; j <= n_transfer; j++) {
        outstream << grid_data[(n_transfer + 1) * i + j] << " ";
      }
      outstream << std::endl;
    }
    outstream.close();
  }
}

inline void ReadGridData(TV& grid_data, Index& grid_N, T& dx, const std::string& filename, const size_t n_transfer = d) {
  std::ifstream instream;
  instream.open(filename);
  if (!instream) {
    std::cerr << "Unable to open file " << filename << std::endl;
    exit(1);  // call system to stop
  }
  std::string line;
  getline(instream, line);
  std::stringstream ss(line);
  ss >> grid_N[0] >> grid_N[1];

  #ifndef TWO_D
  ss >> grid_N[2];
  #endif
  ss >> dx;
  sz flat_index = 0;
  sz num_grid = grid_N[0] * grid_N[1];
  #ifndef TWO_D
  num_grid *= grid_N[2];
  #endif
  grid_data.resize((n_transfer + 1) * num_grid);
  while (!instream.eof()) {
    getline(instream, line);
    std::stringstream ss(line);
    for (size_t j = 0; j <= n_transfer; j++)
      ss >> grid_data[(n_transfer + 1) * flat_index + j];
    flat_index++;
  }
}

inline void WriteGridData(const TV& phi, const Grid& grid, const std::string& filename) {
  #ifndef TWO_D
  TGSLAssert(false, "3D grid visualization not available");
  #else
  std::ofstream outstream(filename);
  if (outstream.good()) {
    size_t num_grid = grid.grid_N[0] * grid.grid_N[1];
    outstream << "# " << num_grid << " points\n";
    for (int i = 0; i < grid.grid_N[0]; i++) {
      for (int j = 0; j < grid.grid_N[1]; j++) {
        Index index = {i, j};
        nm flat_index = grid.FlatIndex(index);
        Particle grid_pos = grid.Node(index);
        outstream << "v " << grid_pos[0] << " " << grid_pos[1] << " " << phi[flat_index] << std::endl;
      }
    }
    outstream.close();
  }
  #endif
}

inline void WriteGEO(const TVP& positions, const sz Np, const std::string& filename) {
  std::ofstream outstream(filename);
  if (outstream.good()) {
    outstream << "PGEOMETRY V5"
              << "\n";
    outstream << "NPoints " << Np << " NPrims 0"
              << "\n";
    outstream << "NPointGroups 0 NPrimGroups 0"
              << "\n";
    outstream << "NPointAttrib 0 NVertexAttrib 0 NPrimAttrib 0 NAttrib 0"
              << "\n";

    // Write points
    for (sz i = 0; i < Np; i++) {
      for (size_t j = 0; j < 3; j++) {
        float val = (d == 3 || j != 2) ? float(positions[i][j]) : 0.f;
        outstream << val << " ";
      }
      outstream << 1 << "\n";
    }

    outstream << "beginExtra"
              << "\n";
    outstream << "endExtra"
              << "\n";

    outstream.close();
  } else {
    std::cout << "Could not open file for writing: " << filename << std::endl;
  }
}

inline void ReadGEO(const std::string& filename, TVP& positions) {
  std::ifstream instream(filename);
  if (instream.good()) {
    // Read the header
    std::string header_line;
    std::getline(instream, header_line); // Read Magic Number (and discard)
    header_line.clear();

    std::getline(instream, header_line); // NPoints # NPrims #
    std::vector<std::string> line_2(4);
    std::stringstream stream(header_line);
    for (size_t i = 0; i < 4 && stream.good(); ++i)
      stream >> line_2[i];
    sz npoints = sz(std::stoi(line_2[1]));
    sz nprims = sz(std::stoi(line_2[3]));
    TGSLAssert(nprims == 0, "IO::ReadGEO: This version of read geo should have 0 primitives.");
    header_line.clear();

    std::getline(instream, header_line); // NPointGroups # NPrimGroups #
    std::vector<std::string> line_3(4);
    stream = std::stringstream(header_line);
    for (size_t i = 0; i < 4 && stream.good(); ++i)
      stream >> line_3[i];
    TGSLAssert(std::stoi(line_3[1]) == 0 && std::stoi(line_3[3]) == 0, "IO::ReadGEO: NPointGroups and NPrimGroups not currently supported.");
    header_line.clear();

    std::getline(instream, header_line); // NPointAttrib 0 NVertexAttrib 0 NPrimAttrib 0 NAttrib 0
    std::vector<std::string> line_4(8);
    stream = std::stringstream(header_line);
    for (size_t i = 0; i < 8 && stream.good(); ++i)
      stream >> line_4[i];
    TGSLAssert(std::stoi(line_4[1]) == 0 && std::stoi(line_4[3]) == 0 && std::stoi(line_4[5]) == 0 && std::stoi(line_4[7]) == 0, "IO::ReadGEO: nPointAttrib, NVertexAttrib, nPrimAttrib, and NAttrib not currently supported.");
    header_line.clear();

    positions.resize(npoints);
    T w;
    #ifdef TWO_D
    T ignore;
    #endif
    for (sz i = 0; i < npoints; ++i) {
      #ifdef TWO_D
      for (size_t l = 0; l < 2; ++l)
        instream >> positions[i][l];
      instream >> ignore;
      instream >> w;
      TGSLAssert(w == T(1), "IO::ReadGEO: Coordinate w should be 1.");
      #else
      for (size_t l = 0; l < 3; ++l)
        instream >> positions[i][l];
      instream >> w;
      TGSLAssert(w == T(1), "IO::ReadGEO: Coordinate w should be 1.");
      #endif
    }
  }
}

inline void WriteGEO(const TVP& positions, const sz Np, const IV& mesh, const sz mesh_size, const std::string& filename) {
  std::ofstream outstream(filename);
  if (outstream.good()) {
    #ifdef TWO_D
    sz nPrims = mesh_size / 3;
    #else
    sz nPrims = mesh_size;
    #endif
    outstream << "PGEOMETRY V5"
              << "\n";
    outstream << "NPoints " << Np << " NPrims " << nPrims << "\n";
    outstream << "NPointGroups 0 NPrimGroups 0"
              << "\n";
    outstream << "NPointAttrib 0 NVertexAttrib 0 NPrimAttrib 0 NAttrib 0"
              << "\n";

    // Write points
    for (sz i = 0; i < Np; i++) {
      for (size_t j = 0; j < 3; j++) {
        float val = (d == 3 || j != 2) ? float(positions[i][j]) : 0.f;
        outstream << val << " ";
      }
      outstream << 1 << "\n";
    }

    // Write poly
    #ifdef TWO_D
    for (sz i = 0; i < mesh_size / 3; i++) {
      outstream << "Poly 3 < ";
      for (size_t j = 0; j < 3; j++) {
        outstream << mesh[3 * i + j] << " ";
      }
      outstream << "\n";
    }
    #else
    for (sz i = 0; i < mesh_size / 4; i++) {
      outstream << "Poly 3 < ";
      outstream << mesh[4 * i] << " " << mesh[4 * i + 1] << " " << mesh[4 * i + 2] << "\n";
      outstream << "Poly 3 < ";
      outstream << mesh[4 * i] << " " << mesh[4 * i + 2] << " " << mesh[4 * i + 3] << "\n";
      outstream << "Poly 3 < ";
      outstream << mesh[4 * i] << " " << mesh[4 * i + 3] << " " << mesh[4 * i + 1] << "\n";
      outstream << "Poly 3 < ";
      outstream << mesh[4 * i + 1] << " " << mesh[4 * i + 3] << " " << mesh[4 * i + 2] << "\n";
    }
    #endif

    outstream << "beginExtra"
              << "\n";
    outstream << "endExtra"
              << "\n";

    outstream.close();
  } else {
    std::cout << "Could not open file for writing: " << filename << std::endl;
  }
}

inline void WriteTrisGEO(const TVP& positions, const IV& mesh, const TV& activation, const std::string& filename) {
  std::ofstream outstream(filename);
  if (outstream.good()) {
    TGSLAssert(activation.size() == mesh.size()/3, "IO::WriteTrisGEO: activation is not sized the same as mesh.");
    sz nPrims = mesh.size() / 3;
    outstream << "PGEOMETRY V5"
              << "\n";
    outstream << "NPoints " << positions.size() << " NPrims " << nPrims << "\n";
    outstream << "NPointGroups 0 NPrimGroups 0"
              << "\n";
    outstream << "NPointAttrib 0 NVertexAttrib 0 NPrimAttrib 1 NAttrib 0"
              << "\n";

    // Write points
    for (sz i = 0; i < positions.size(); i++) {
      for (size_t j = 0; j < 3; j++) {
        float val = (d == 3 || j != 2) ? float(positions[i][j]) : 0.f;
        outstream << val << " ";
      }
      outstream << 1 << "\n";
    }
    outstream << "PrimitiveAttrib\n" << "activation 1 float 0\n";
    // Write poly
    for (sz i = 0; i < mesh.size()/3; i++) {
      outstream << "Poly 3 < ";
      outstream << mesh[3 * i] << " " << mesh[3 * i + 1] << " " << mesh[3 * i + 2] << " [" << activation[i] << "]" << "\n";
    }

    outstream << "beginExtra"
              << "\n";
    outstream << "endExtra"
              << "\n";

    outstream.close();
  } else {
    std::cout << "Could not open file for writing: " << filename << std::endl;
  }
}

inline void WriteTrisGEO(const TVP& positions, const sz Np, const IV& mesh, const sz mesh_size, const std::string& filename) {
  std::ofstream outstream(filename);
  if (outstream.good()) {
    sz nPrims = mesh_size / 3;
    outstream << "PGEOMETRY V5"
              << "\n";
    outstream << "NPoints " << Np << " NPrims " << nPrims << "\n";
    outstream << "NPointGroups 0 NPrimGroups 0"
              << "\n";
    outstream << "NPointAttrib 0 NVertexAttrib 0 NPrimAttrib 0 NAttrib 0"
              << "\n";

    // Write points
    for (sz i = 0; i < Np; i++) {
      for (size_t j = 0; j < 3; j++) {
        float val = (d == 3 || j != 2) ? float(positions[i][j]) : 0.f;
        outstream << val << " ";
      }
      outstream << 1 << "\n";
    }

    // Write poly
    for (sz i = 0; i < mesh_size/3; i++) {
      outstream << "Poly 3 < ";
      outstream << mesh[3 * i] << " " << mesh[3 * i + 1] << " " << mesh[3 * i + 2] << "\n";
    }

    outstream << "beginExtra"
              << "\n";
    outstream << "endExtra"
              << "\n";

    outstream.close();
  } else {
    std::cout << "Could not open file for writing: " << filename << std::endl;
  }
}

inline void WriteTrisGEO(const TVP& positions, const std::string& filename) {
  IV mesh;
  mesh.resize(positions.size());
  for(sz i=0;i<mesh.size();++i)
    mesh[i] = static_cast<TGSL::nm>(i);

  WriteTrisGEO(positions, positions.size(), mesh, mesh.size(), filename);
}

inline void WriteSegsGEO(const TVP& positions, const sz Np, const IV& mesh, const sz mesh_size, const std::string& filename) {
  #ifndef TWO_D
  std::ofstream outstream(filename);
  if (outstream.good()) {
    sz nPrims = mesh_size / 2;
    outstream << "PGEOMETRY V5"
              << "\n";
    outstream << "NPoints " << Np << " NPrims " << nPrims << "\n";
    outstream << "NPointGroups 0 NPrimGroups 0"
              << "\n";
    outstream << "NPointAttrib 0 NVertexAttrib 0 NPrimAttrib 0 NAttrib 0"
              << "\n";

    // Write points
    for (sz i = 0; i < Np; i++) {
      for (size_t j = 0; j < 3; j++) {
        outstream << float(positions[i][j]) << " ";
      }
      outstream << 1 << "\n";
    }

    // Write poly
    for (sz i = 0; i < mesh_size/2; i++) {
      outstream << "Poly 2 < ";
      outstream << mesh[2 * i] << " " << mesh[2 * i + 1] << "\n";
    }

    outstream << "beginExtra"
              << "\n";
    outstream << "endExtra"
              << "\n";

    outstream.close();
  } else {
    std::cout << "Could not open file for writing: " << filename << std::endl;
  }
  #else
  std::ofstream outstream(filename);
  if (outstream.good()) {
    sz nPrims = mesh_size / 2;
    outstream << "PGEOMETRY V5"
              << "\n";
    outstream << "NPoints " << Np << " NPrims " << nPrims << "\n";
    outstream << "NPointGroups 0 NPrimGroups 0"
              << "\n";
    outstream << "NPointAttrib 0 NVertexAttrib 0 NPrimAttrib 0 NAttrib 0"
              << "\n";

    // Write points
    for (sz i = 0; i < Np; i++) {
      for (size_t j = 0; j < 2; j++) {
        float val = (d == 2 || j != 2) ? float(positions[i][j]) : 0.f;
        outstream << val << " ";
      }
      outstream << 0 << " " << 1 << "\n";
    }

    // Write poly
    for (sz i = 0; i < mesh_size/2; i++) {
      outstream << "Poly 2 < ";
      outstream << mesh[2 * i] << " " << mesh[2 * i + 1] << "\n";
    }

    outstream << "beginExtra"
              << "\n";
    outstream << "endExtra"
              << "\n";

    outstream.close();
  } else {
    std::cout << "Could not open file for writing: " << filename << std::endl;
  }
  #endif
}

inline void ReadTrisGEO(const std::string& filename, TVP& positions, IV& mesh) {
  std::ifstream instream(filename);
  if (instream.good()) {
    // Read the header
    std::string header_line;
    std::getline(instream, header_line); // Read Magic Number (and discard)
    header_line.clear();

    std::getline(instream, header_line); // NPoints # NPrims #
    std::vector<std::string> line_2(4);
    std::stringstream stream(header_line);
    for (size_t i = 0; i < 4 && stream.good(); ++i)
      stream >> line_2[i];
    sz npoints = sz(std::stoi(line_2[1]));
    sz nprims = sz(std::stoi(line_2[3]));
    header_line.clear();

    std::getline(instream, header_line); // NPointGroups # NPrimGroups #
    std::vector<std::string> line_3(4);
    stream = std::stringstream(header_line);
    for (size_t i = 0; i < 4 && stream.good(); ++i)
      stream >> line_3[i];
    TGSLAssert(std::stoi(line_3[1]) == 0 && std::stoi(line_3[3]) == 0, "IO::ReadTrisGEO: NPointGroups and NPrimGroups not currently supported.");
    header_line.clear();

    std::getline(instream, header_line); // NPointAttrib 0 NVertexAttrib 0 NPrimAttrib 0 NAttrib 0
    std::vector<std::string> line_4(8);
    stream = std::stringstream(header_line);
    for (size_t i = 0; i < 8 && stream.good(); ++i)
      stream >> line_4[i];
    TGSLAssert(std::stoi(line_4[1]) == 0 && std::stoi(line_4[3]) == 0 && std::stoi(line_4[5]) == 0 && std::stoi(line_4[7]) == 0, "IO::ReadTrisGEO: nPointAttrib, NVertexAttrib, nPrimAttrib, and NAttrib not currently supported.");
    header_line.clear();

    positions.resize(npoints);
    T w;
    std::string ignore;
    for (sz i = 0; i < npoints; ++i) {
      #ifdef TWO_D
      for (size_t l = 0; l < 2; ++l)
        instream >> positions[i][l];
      instream >> ignore;
      instream >> w;
      TGSLAssert(w == T(1), "IO::ReadGEO: Coordinate w should be 1.");
      #else
      for (size_t l = 0; l < 3; ++l)
        instream >> positions[i][l];
      instream >> w;
      TGSLAssert(w == T(1), "IO::ReadGEO: Coordinate w should be 1.");
      #endif
    }

    mesh.resize(3 * nprims);
    for (sz i = 0; i < nprims; ++i) {
      for (size_t l = 0; l < 3; ++l)
        instream >> ignore;
      for (size_t l = 0; l < 3; ++l)
        instream >> mesh[3 * i + l];
    }
  }
}

inline void WriteQuadMeshGEO(const TVP& positions, const sz Np, const IV& mesh, const sz mesh_size, const std::string& filename, sz offset = 0) {
  #ifndef TWO_D

  // TGSLAssert(false,"IO::WriteQuadMeshGEO is only defined for 2D.");
  // Yushan: Commented out for writing quad boundary of hex mesh in 3D
  #endif
  std::ofstream outstream(filename);
  if (outstream.good()) {
    sz nPrims = mesh_size / 4;
    outstream << "PGEOMETRY V5"
              << "\n";
    outstream << "NPoints " << Np << " NPrims " << nPrims << "\n";
    outstream << "NPointGroups 0 NPrimGroups 0"
              << "\n";
    outstream << "NPointAttrib 0 NVertexAttrib 0 NPrimAttrib 0 NAttrib 0"
              << "\n";

    // Write points
    for (sz i = 0; i < Np; i++) {
      for (size_t j = 0; j < 3; j++) {
        float val = (d == 3 || j != 2) ? float(positions[i][j]) : 0.f;
        outstream << val << " ";
      }
      outstream << 1 << "\n";
    }

    // Write poly
    for (sz i = 0; i < mesh_size / 4; i++) {
      outstream << "Poly 4 < ";
      for (size_t j = 0; j < 4; j++) {
        outstream << mesh[4 * i + j] - offset << " ";
      }
      outstream << "\n";
    }

    outstream << "beginExtra"
              << "\n";
    outstream << "endExtra"
              << "\n";

    outstream.close();
  } else {
    std::cout << "Could not open file for writing: " << filename << std::endl;
  }
}

inline void WriteHexMeshGEO(const TVP& positions, const sz Np, const IV& mesh, const sz mesh_size, const std::string& filename, sz offset = 0) {
  #if defined ONE_D || defined TWO_D
  TGSLAssert(false,"IO::WriteHexMeshGEO is only defined for 3D.");
  #else
  std::ofstream outstream(filename);
  TGSLAssert(mesh_size % 8 == 0, "mesh size wrong, not divisible by 8");
  if (outstream.good()) {
    sz nPrims = (mesh_size / 8)*6;
    outstream << "PGEOMETRY V5"
              << "\n";
    outstream << "NPoints " << Np << " NPrims " << nPrims << "\n";
    outstream << "NPointGroups 0 NPrimGroups 0"
              << "\n";
    outstream << "NPointAttrib 0 NVertexAttrib 0 NPrimAttrib 0 NAttrib 0"
              << "\n";

    // Write points
    for (sz i = 0; i < Np; i++) {
      for (size_t j = 0; j < 3; j++) {
        float val = static_cast<float>(positions[i][j]);
        outstream << val << " ";
      }
      outstream << 1 << "\n";
    }

    // Write poly
    IVV hex_faces = {{0,3,2,1}, {4,5,6,7}, {5,1,2,6}, {0,1,5,4}, {2,3,7,6}, {0,4,7,3}};
    for (sz i = 0; i < mesh_size / 8; i++) {
      for (sz j = 0; j< 6; j++){
        outstream << "Poly 4 < ";
        for (sz k = 0; k< 4; k++){
          outstream << mesh[8*i+hex_faces[j][k]] - offset << " ";
        }
        outstream << "\n";
      }
    }

    outstream << "beginExtra"
              << "\n";
    outstream << "endExtra"
              << "\n";

    outstream.close();
  } else {
    std::cout << "Could not open file for writing: " << filename << std::endl;
  }
  #endif
}

inline void WriteCubeMeshOBJ(const TVP& positions, const sz Np, const IV& mesh, const sz mesh_size, const std::string& filename) {
  #if defined ONE_D || defined TWO_D
  TGSLAssert(false,"IO::WriteHexMeshObj is only defined for 3D.");
  #else
  std::ofstream outstream(filename);
  if (outstream.good()) {
    // Write points
    for (sz i = 0; i < Np; i++) {
      outstream << "v ";
      for (size_t j = 0; j < 3; j++) {
        float val = float(positions[i][j]);
        outstream << val << (j == 2 ? "\n" : " ");
      }
    }

    // Write faces
    for (sz i = 0; i < mesh_size / 8; i++) {
      outstream << "f " << mesh[8 * i + 0] + 1 << " " << mesh[8 * i + 3] + 1 << " " << mesh[8 * i + 2] + 1 << " " << mesh[8 * i + 1] + 1 << "\n";
      outstream << "f " << mesh[8 * i + 4] + 1 << " " << mesh[8 * i + 5] + 1 << " " << mesh[8 * i + 6] + 1 << " " << mesh[8 * i + 7] + 1 << "\n";
      outstream << "f " << mesh[8 * i + 0] + 1 << " " << mesh[8 * i + 1] + 1 << " " << mesh[8 * i + 5] + 1 << " " << mesh[8 * i + 4] + 1 << "\n";
      outstream << "f " << mesh[8 * i + 2] + 1 << " " << mesh[8 * i + 3] + 1 << " " << mesh[8 * i + 7] + 1 << " " << mesh[8 * i + 6] + 1 << "\n";
      outstream << "f " << mesh[8 * i + 0] + 1 << " " << mesh[8 * i + 4] + 1 << " " << mesh[8 * i + 7] + 1 << " " << mesh[8 * i + 3] + 1 << "\n";
      outstream << "f " << mesh[8 * i + 1] + 1 << " " << mesh[8 * i + 2] + 1 << " " << mesh[8 * i + 6] + 1 << " " << mesh[8 * i + 5] + 1 << "\n";
    }

    outstream.close();
  } else {
    std::cout << "Could not open file for writing: " << filename << std::endl;
  }
  #endif
}

inline void WriteTXTScalarField(const TVP& x, const TV& field, const T& time, const std::string& filename) {
  std::ofstream outstream(filename);
  if (outstream.good()) {
    for (size_t i = 0; i < x.size(); ++i) {
      outstream << time << " ";
      for (size_t j = 0; j < d; ++j)
        outstream << x[i][j] << " ";
      outstream << field[i];
      outstream << std::endl;
    }

    outstream.close();
  }
}

inline void WriteTXTFieldGrid(const Grid& grid, const TV& grid_data, const size_t& start_offset, const size_t& end_offset, const size_t& n_transfer, const T& time, const std::string& filename) {
  sz grid_N_prod = grid.grid_N[0];
  for (size_t dd = 1; dd < d; ++dd)
    grid_N_prod *= grid.grid_N[dd];
  std::vector<TV> extracted;
  TVP x;
  IVI indices;
  for(sz i = 0; i < grid_N_prod; ++i){
    x.emplace_back(grid.Node(static_cast<nm>(i)));
    Index mi; grid.Lin2MultiIndex(static_cast<nm>(i), mi);
    indices.emplace_back(mi);
    TV ei;
    for(sz j = i*(n_transfer+1) + 1 + start_offset; j <= i*(n_transfer+1) + 1 + end_offset; ++j){
      ei.emplace_back(grid_data[j]);
    }
    extracted.emplace_back(ei);
  }

  std::ofstream outstream(filename);
  if (outstream.good()) {
    for (sz i = 0; i < x.size(); ++i) {
      outstream << time << " ";
      outstream << i << " ";
      for (size_t j = 0; j < d; ++j)
        outstream << indices[i][j] << " ";
      for (size_t j = 0; j < d; ++j)
        outstream << x[i][j] << " ";
      for (size_t j = 0; j < (end_offset-start_offset); ++j)
        outstream << extracted[i][j] << " ";
      outstream << extracted[i][(end_offset-start_offset)];
      outstream << std::endl;
    }

    outstream.close();
  }
}

inline void WriteTXTFieldGrid(const Grid& grid, const TV& grid_data, const size_t& offset, const size_t& n_transfer, const T& time, const std::string& filename) {
  WriteTXTFieldGrid(grid, grid_data, offset, offset, n_transfer, time, filename);
}

template <typename VectorType>
inline void WriteTXTVectorField(const TVP& x, const VectorType& field, const T& time, const std::string& filename) {
  std::ofstream outstream(filename);
  if (outstream.good()) {
    for (size_t i = 0; i < x.size(); ++i) {
      outstream << time << " ";
      for (size_t j = 0; j < d; ++j)
        outstream << x[i][j] << " ";
      for (size_t j = 0; j < field[i].size() - 1; ++j)
        outstream << field[i][j] << " ";
      outstream << field[i][field[i].size() - 1];
      outstream << std::endl;
    }

    outstream.close();
  }
}

class Sensors {
 public:

  IVV p_sensors;          // For each sensor, vector of particles used as sensors
  IV id_sensors;          // For each particle, sensor for which they account (if any)

  Sensors(){}
  Sensors(const T Np, const TVP &sensors_pos, const TVP &x, const T radius = 0.005) {
    std::cout << "Finding sensor particles... ";

    p_sensors.resize(sensors_pos.size());
    id_sensors.resize(static_cast<size_t>(Np), 0);   // 0 = not a sensor, 1 = sensor

    #pragma omp parallel for
    for (nm s = 0; s < (nm)sensors_pos.size(); s++) {
      for (size_t p = 0; p < Np; p++) {
        T dist = T(0);
        for (size_t i = 0; i < d; i++)
          dist += (sensors_pos[s][i] -x[p][i]) * (sensors_pos[s][i] - x[p][i]);
        dist = sqrt(dist);
        if (dist <= radius)
          p_sensors[s].push_back(static_cast<TGSL::nm>(p));
      }
    }

    std::cout << "Done." << std::endl;

    T s_count = T(0);
    for (size_t s = 0; s < p_sensors.size(); s++)
      s_count += p_sensors[s].size();
    std::cout << GREEN << "Number of sensors: " << s_count << RESET << std::endl;

    for (size_t s = 0; s < p_sensors.size(); s++)
      for (size_t p = 0; p < p_sensors[s].size(); p++)
        id_sensors[p_sensors[s][p]] = 1;
  }

  void WriteSensorsData(const std::string &filename, const T dt, const TVP &x, const TVP &v, const TVV &a, const TV &F_e) {
    /*
      2D Format:
        dt
        Sensor id - # of particle sensors
        x, x, v, v, a, a, F, F, F, F for particle sensor 1
        x, x, v, v, a, a, F, F, F, F for particle sensor 2
        ...
      3D Format:
        dt
        Sensor id - # of particle sensors
        x, x, x, v, v, v, a, a, a, F, F, F, F, F, F, F, F, F for particle sensor 1
        x, x, x, v, v, v, a, a, a, F, F, F, F, F, F, F, F, F for particle sensor 2
        ...
    */

    std::ofstream output; output.open(filename);
    output << std::to_string(dt) + "\n";
    for (size_t s = 0; s < p_sensors.size(); s++) {
      output << std::to_string(s + 1) + " " + std::to_string(p_sensors[s].size()) + "\n";
      for (size_t p = 0; p < p_sensors[s].size(); p++) {
        nm p_id = p_sensors[s][p];
        for (size_t i = 0; i < d; i++)
          output << std::setprecision(12) << std::to_string(x[p_id][i]) + ", ";
        for (size_t i = 0; i < d; i++)
          output << std::setprecision(12) << std::to_string(v[p_id][i]) + ", ";
        for (size_t i = 0; i < d; i++)
          output << std::setprecision(12) << std::to_string(a[p_id][i]) + ", ";
        for (size_t r = 0; r < d; ++r) {
          for (size_t c = 0; c < d; ++c) {
            size_t index = d * d * p_id + d * r + c;
            output << std::setprecision(12) << std::to_string(F_e[index]) + ", ";
          }
        }
        output << "\n";
      }
    }
    output.close();
  }

  void WriteSensorsData(const std::string &filename, const T dt, const TVP &x, const TVP &v, const TV &F_e) {
    TVV a(x.size(), Vector());
    WriteSensorsData(filename, dt, x, v, a, F_e);
  }

  void WriteSensorsDataAverage(const std::string &output_dir, const T dt, const TVP &x, const TVP &v) {

    for (size_t s = 0; s < p_sensors.size(); s++) {

      std::string filename_x = output_dir + "/position" + std::to_string(s+1) + ".txt";
      std::string filename_v = output_dir + "/velocity" + std::to_string(s+1) + ".txt";

      std::ofstream output_x; output_x.open(filename_x, std::ios_base::app);
      std::ofstream output_v; output_v.open(filename_v, std::ios_base::app);

      Particle x_av = Particle(), v_av = Particle();
      for (size_t p = 0; p < p_sensors[s].size(); p++) {
        nm p_id = p_sensors[s][p];
        for (size_t i = 0; i < d; i++) {
          x_av[i] += x[p_id][i];
          v_av[i] += v[p_id][i];
        }
      }
      for (size_t i = 0; i < d; i++) {
        x_av[i] /= p_sensors[s].size();
        v_av[i] /= p_sensors[s].size();

        output_x << std::setprecision(12) << std::to_string(x_av[i]) + ", ";
        output_v << std::setprecision(12) << std::to_string(v_av[i]) + ", ";
      }

      output_x << "\n";
      output_v << "\n";
    }

    std::string filename_dt = output_dir + "/dt.txt";
    std::ofstream output_dt; output_dt.open(filename_dt, std::ios_base::app);
    output_dt << std::to_string(dt) + "\n";
  }
};

namespace JSON {

inline std::istream& operator>>(std::istream& in, std::pair<TGSL::Particle,TGSL::Vector>& value)
{
    /* Transforms from Maya look like this:
    {
    	"FACIAL_C_Jaw": [
    		[
    			-0.04815200828336853,
    			144.4854436613277,
    			2.91372715526443
    		],
    			[
    				40.18255995024729,
    				-0.14419194738208552,
    				0.2029523041105289
    			]
    	]
    }
    */
    char ch;
    in  >> ch >> ch
    	>> value.first[0] >> ch >> value.first[1] >> ch >> value.first[2]
    	>> ch >> ch >> ch
    	>> value.second[0] >> ch >> value.second[1] >> ch >> value.second[2]
    	>> ch >> ch;
    return in;
}

template <class KEY, class VALUE>
bool ReadSimpleJSON(const std::string& filename, std::unordered_map<KEY, VALUE>& data, std::vector<KEY>* ordering=nullptr)
{
    if(!std::filesystem::exists(filename))
    {
        std::cout << "File not found: '" << filename << "'" << std::endl;
        return false;
    }
    std::shared_ptr<std::istream> infile(new std::ifstream(filename, std::ios::in));
    if (!infile)
    {
        std::cout << "Failed to open JSON file: '" << filename << "'" << std::endl;
        return false;
    }
    else
        std::cout << "Reading JSON file: '" << filename << "'" << std::endl;
    std::istream& infile_ref = *infile;
    std::string buffer;
    buffer.reserve(128);
    while (infile_ref.good())
    {
        infile_ref >> std::ws;
        const int next_ch = infile_ref.peek();
        if (next_ch == '{')
        {
            std::getline(infile_ref, buffer); // discard until \n
        }
        else if (next_ch == '}' || next_ch == std::char_traits<char>::eof())
        {
            break;
        }
        else
        {
            KEY key;
            VALUE val;
            infile_ref >> key;
            infile_ref >> val;

            // Remove trailing colon
            if (key.back() == ':')
                key = key.substr(0, key.size() - 1);
            // Remove quotes
            if (key.front() == '"' && key.back() == '"')
                key = key.substr(1, key.size() - 2);
            data[key] = val;
            if(ordering)
                ordering->push_back(key);

            std::getline(infile_ref, buffer); // discard until \n
        }
    }
    return infile_ref.good() || infile_ref.eof();
}

template <class KEY, class VALUE>
void WriteSimpleJSON(const std::string& filename, const std::unordered_map<KEY, VALUE>& data, const std::vector<KEY>* ordering=nullptr)
{
    std::shared_ptr<std::ostream> outfile(new std::ofstream(filename)); // ascii
    if(!outfile)
    {
        std::cout << "Failed to write JSON file: '" << filename << "'" << std::endl;
        return;
    }
    std::ostream& outfile_ref = *outfile;
    outfile_ref << "{" << std::endl;
    if(ordering)
    {
        const auto dataItEnd = data.end();
        for(auto itBegin=ordering->begin(), it=ordering->begin(), itEnd=ordering->end(); it != itEnd; ++it)
        {
            const KEY& key = *it;
            if(it != itBegin)
                outfile_ref << "," << std::endl;
            const auto dataIt = data.find(key);
            if(dataIt == dataItEnd)
            {
                std::cout << "Ordering key '" << key << "' not found in data map!" << std::endl;
                exit(1);
            }
            outfile_ref
                << std::fixed << std::setprecision(11) << std::setfill('0')
                << "    \"" << key << "\": " << dataIt->second;
        }
    }
    else
    {
        for(auto itBegin=data.begin(), it=data.begin(), itEnd=data.end(); it != itEnd; ++it)
        {
            if(it != itBegin)
                outfile_ref << "," << std::endl;
            outfile_ref
                << std::fixed << std::setprecision(11) << std::setfill('0')
                << "    \"" << it->first << "\": " << it->second;
        }
    }
    outfile_ref << std::endl << "}" << std::endl;
    outfile.reset();
    std::cout << "Wrote file '" << filename << "'." << std::endl;
}


} // namespace JSON
} // namespace IO

