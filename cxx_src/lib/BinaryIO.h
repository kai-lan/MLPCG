#ifndef BINARY_IO_H
#define BINARY_IO_H

//#include "AlembicIO.h"
//#include "ZIP.h"


#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <Definitions.h>
#include <Filesystem.h>

namespace IO {

inline void Serialize(const nm& v, const std::string& filename) {
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  out.write((char*)&v, sizeof(nm));
  out.close();
}

inline void Serialize(const sz& v, const std::string& filename) {
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  out.write((char*)&v, sizeof(sz));
  out.close();
}

inline void Serialize(const T& v, const std::string& filename) {
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  out.write((char*)&v, sizeof(T));
  out.close();
}

inline void Serialize(const std::vector<std::unordered_map<nm, nm>>& v, const std::string& filename) {
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  size_t v_size = v.size();
  out.write((char*)&v_size, sizeof(size_t));

  for (auto m : v) {
    size_t m_size = m.size();
    out.write((char*)&m_size, sizeof(size_t));
    for (auto x : m) {
      out.write((char*)&(x.first), sizeof(nm));
      out.write((char*)&(x.second), sizeof(nm));
    }
  }

  out.close();
}

template <typename T>
inline void Serialize(const std::vector<T>& v, const std::string& filename) {
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  size_t v_size = v.size();
  out.write((char*)&v_size, sizeof(size_t));
  out.write((char*)&v[0], v_size * sizeof(T));
  out.close();
}

template <typename T>
inline void Serialize(const std::vector<T>& v, const sz n, const std::string& filename) {
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  out.write((char*)&n, sizeof(size_t));
  out.write((char*)&v[0], n * sizeof(T));
  out.close();
}

template <typename T>
inline void Serialize(const std::vector<std::vector<T>>& v, const std::string& filename) {
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  size_t v_size = v.size();
  out.write((char*)&v_size, sizeof(size_t) );
  for (auto m : v) {
    size_t m_size = m.size();
    out.write((char*)&m_size, sizeof(size_t));
    for (auto x : m) {
      out.write((char*)&x, sizeof(T));
    }
  }
  out.close();
}

inline void Serialize(const std::unordered_map<nm, nm>& v, const std::string& filename) {
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  size_t v_size = v.size();
  out.write((char*)&v_size, sizeof(size_t));
  for (auto x : v) {
    out.write((char*)&(x.first), sizeof(nm));
    out.write((char*)&(x.second), sizeof(nm));
  }
  out.close();
}


inline void Deserialize(nm& v, const std::string& filename) {
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  in.read((char*)&v, sizeof(nm));
  in.close();
}

inline void Deserialize(sz& v, const std::string& filename) {
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  in.read((char*)&v, sizeof(sz));
  in.close();
}

inline void Deserialize(T& v, const std::string& filename) {
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  in.read((char*)&v, sizeof(T));
  in.close();
}

inline void Deserialize(std::vector<std::unordered_map<nm, nm>>& v, const std::string& filename) {
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  size_t v_size;
  in.read((char*)&v_size, sizeof(size_t));
  v.resize(v_size);

  for (size_t i = 0; i < v_size; ++i) {
    size_t m_size;
    in.read((char*)&m_size, sizeof(size_t));
    v[i].reserve(m_size);
    for (size_t j = 0; j < m_size; ++j) {
      nm key, value;
      in.read((char*)&key, sizeof(nm));
      in.read((char*)&value, sizeof(nm));
      v[i][key] = value;
    }
  }

  in.close();
}

template <typename T>
inline void Deserialize(std::vector<T>& v, const std::string& filename) {
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  size_t v_size;
  in.read((char*)&v_size, sizeof(size_t));
  v.resize(v_size);
  in.read((char*)&v[0], v_size * sizeof(T));
  in.close();
}

inline void Deserialize(std::unordered_map<nm, nm>& v, const std::string& filename) {
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  size_t v_size;
  in.read((char*)&v_size, sizeof(size_t));
  v.reserve(v_size);
  for (size_t i = 0; i < v_size; ++i) {
    nm key, value;
    in.read((char*)&key, sizeof(nm));
    in.read((char*)&value, sizeof(nm));
    v[key] = value;
  }
  in.close();
}

template <typename T>
inline void Deserialize(std::vector<std::vector<T>>& v, const std::string& filename) {
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  size_t v_size;
  in.read((char*)&v_size, sizeof(size_t));
  v.resize(v_size);
  for (size_t i = 0; i < v_size; ++i) {
    size_t m_size;
    in.read((char*)&m_size, sizeof(size_t));
    v[i].resize(m_size);
    for (size_t j = 0; j < m_size; ++j) {
      T value;
      in.read((char*)&value, sizeof(T));
      v[i][j] = value;
    }
  }
  in.close();
}


namespace Eigen {

using namespace std;
using namespace ::Eigen;

template <typename T>
void Serialize(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v, const std::string& filename) {
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  size_t v_size = v.size();
  out.write((char*)&v_size, sizeof(size_t));
  out.write((char*)v.data(), v_size * sizeof(T));
  out.close();
}

template <typename T>
void Deserialize(Eigen::Matrix<T, Eigen::Dynamic, 1>& v, const std::string& filename) {
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  size_t v_size;
  in.read((char*)&v_size, sizeof(size_t));
  v.resize(v_size);
  in.read((char*)v.data(), v_size * sizeof(T));
  in.close();
}

// Adapted from https://scicomp.stackexchange.com/a/21438/32904
// Why isn't this built into Eigen?

template <typename T, int OptionsBitFlag, typename Index>
void Serialize(SparseMatrix<T, OptionsBitFlag, Index>& m, const std::string& filename) {
  typedef Eigen::Triplet<T, Index> Trip;

  std::vector<Trip> res;

  fstream writeFile;
  writeFile.open(filename, ios::binary | ios::out);

  if (writeFile.is_open()) {
    Index rows, cols, nnzs, outS, innS;
    rows = m.rows();
    cols = m.cols();
    nnzs = m.nonZeros();
    outS = m.outerSize();
    innS = m.innerSize();

    writeFile.write((const char*)&(rows), sizeof(Index));
    writeFile.write((const char*)&(cols), sizeof(Index));
    writeFile.write((const char*)&(nnzs), sizeof(Index));
    writeFile.write((const char*)&(outS), sizeof(Index));
    writeFile.write((const char*)&(innS), sizeof(Index));

    writeFile.write((const char*)(m.valuePtr()), sizeof(T) * m.nonZeros());
    writeFile.write((const char*)(m.outerIndexPtr()), sizeof(Index) * m.outerSize());
    writeFile.write((const char*)(m.innerIndexPtr()), sizeof(Index) * m.nonZeros());

    writeFile.close();
  }
}

template <typename T, int OptionsBitFlag, typename Index>
void Deserialize(SparseMatrix<T, OptionsBitFlag, Index>& m, const std::string& filename) {
  fstream readFile;
  readFile.open(filename, ios::binary | ios::in);
  if (readFile.is_open()) {
    Index rows, cols, nnz, inSz, outSz;
    readFile.read((char*)&rows, sizeof(Index));
    readFile.read((char*)&cols, sizeof(Index));
    readFile.read((char*)&nnz, sizeof(Index));
    readFile.read((char*)&outSz, sizeof(Index));
    readFile.read((char*)&inSz, sizeof(Index));

    m.resize(rows, cols);
    m.makeCompressed();
    m.resizeNonZeros(nnz);

    readFile.read((char*)(m.valuePtr()), sizeof(T) * nnz);
    readFile.read((char*)(m.outerIndexPtr()), sizeof(Index) * outSz);
    readFile.read((char*)(m.innerIndexPtr()), sizeof(Index) * nnz);

    m.finalize();
    readFile.close();

  }  // file is open
}

}  // namespace Eigen


// Note: use std::string::ends_with() once C++20 is adopted.
inline bool EndsWith(const std::string& src, const std::string& suffix)
{
    const size_t srcLen = src.size();
    const size_t sufLen = suffix.size();
    return srcLen >= sufLen && src.substr(srcLen-sufLen) == suffix;
}


inline std::shared_ptr<std::ostream> SafeOpenOutput(const std::string& filename, bool binary=true, bool append = false)
{
    std::ofstream* outfile;
    if (!append) {
      outfile = new std::ofstream(filename, std::ios::out | std::ios::trunc | std::ios::binary);
    } else {
      outfile = new std::ofstream(filename, std::ios::out | std::ios::app | std::ios::binary);
    }
    return std::shared_ptr<std::ostream>(outfile);
}


template <class T>
bool
ReadBinary(std::istream& InFile, T& x)
{
    if (!InFile.eof())
    {
      InFile.read((char*)(&x), sizeof(T));
      return true;
    }
    return false;
}

template <class T>
bool
WriteBinary(std::ostream& OutFile, const T& x)
{
    if(OutFile.good())
    {
        //OutFile.write(reinterpret_cast<const char*>(&x), sizeof(T));
      OutFile.write((const char*)(&x), sizeof(T));
        return true;
    }
    return false;
}

}  // namespace IO

#endif

