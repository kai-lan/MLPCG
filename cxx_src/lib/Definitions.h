#include <array>
#include <cmath>
#include <cstdint>
#include <execution>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Headers for error handling & debugging
#include <cassert>
#include <stdexcept>


#ifndef DEFINITIONS_INCLUDED
#define DEFINITIONS_INCLUDED

// #define RESET   "\033[0m"
// #define BLACK   "\033[30m"      /* Black */
// #define RED     "\033[31m"      /* Red */
// #define GREEN   "\033[32m"      /* Green */
// #define YELLOW  "\033[33m"      /* Yellow */
// #define BLUE    "\033[34m"      /* Blue */
// #define MAGENTA "\033[35m"      /* Magenta */
// #define CYAN    "\033[36m"      /* Cyan */
// #define WHITE   "\033[37m"      /* White */
// #define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
// #define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
// #define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
// #define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
// #define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
// #define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
// #define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
// #define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

const char RESET[] = "\033[0m";
const char BLACK[] = "\033[30m";      /* Black */
const char RED[]   = "\033[31m";      /* Red */
const char GREEN[]   = "\033[32m";      /* Green */
const char YELLOW[]  = "\033[33m";      /* Yellow */
const char BLUE[]    = "\033[34m";      /* Blue */
const char MAGENTA[] = "\033[35m";      /* Magenta */
const char CYAN[]    = "\033[36m";      /* Cyan */
const char WHITE[]   = "\033[37m";      /* White */
const char BOLDBLACK[]   = "\033[1m\033[30m";      /* Bold Black */
const char BOLDRED[]     = "\033[1m\033[31m";      /* Bold Red */
const char BOLDGREEN[]   = "\033[1m\033[32m";      /* Bold Green */
const char BOLDYELLOW[]  = "\033[1m\033[33m";      /* Bold Yellow */
const char BOLDBLUE[]    = "\033[1m\033[34m";      /* Bold Blue */
const char BOLDMAGENTA[] = "\033[1m\033[35m";      /* Bold Magenta */
const char BOLDCYAN[]    = "\033[1m\033[36m";      /* Bold Cyan */
const char BOLDWHITE[]   = "\033[1m\033[37m";      /* Bold White */

#ifdef BIGNUM
typedef uint_fast64_t sz;
typedef int_fast64_t nm;
#else
typedef size_t sz;
typedef int nm;
#endif

#ifdef ONE_D
constexpr size_t d = 1;
constexpr size_t pow_2_d = 2;
constexpr size_t pow_3_d = 3;
#elif defined TWO_D
constexpr size_t d = 2;
constexpr size_t pow_2_d = 4;
constexpr size_t pow_3_d = 9;
#else
constexpr size_t d = 3;
constexpr size_t pow_2_d = 8;
constexpr size_t pow_3_d = 27;
#endif

using T = double;

using TV = std::vector<T>;
using IV = std::vector<nm>;
using IVV = std::vector<IV>;
using BV = std::vector<unsigned char>;
using BVV = std::vector<BV>;

using Vector2T = std::array<T, 2>;
using Vector3T = std::array<T, 3>;
using Vector4T = std::array<T, 4>;
using Vector2I = std::array<nm, 2>;
using Vector3I = std::array<nm, 3>;
using Vector4I = std::array<nm, 4>;

enum class Cell : unsigned char { active, exterior, inactive };

template <typename T>
std::ostream& operator<<(typename std::enable_if<std::is_enum<T>::value, std::ostream>::type& stream, const T& e) {
  return stream << static_cast<typename std::underlying_type<T>::type>(e);
}

#ifdef ONE_D
using Particle = std::array<T, 1>;
using Vector = std::array<T, 1>;
using Index = std::array<nm, 1>;
using VertexArray = std::array<bool, 2>;
using NeighborArray = std::array<Cell, 2>;
#elif defined TWO_D
using Particle = std::array<T, 2>;
using Vector = std::array<T, 2>;
using Index = std::array<nm, 2>;
using VertexArray = std::array<bool, 4>;
using NeighborArray = std::array<Cell, 4>;
#else
using Particle = std::array<T, 3>;
using Vector = std::array<T, 3>;
using Index = std::array<nm, 3>;
using VertexArray = std::array<bool, 8>;
using NeighborArray = std::array<Cell, 6>;
#endif

using TVP = std::vector<Particle>;
using TVP2 = std::vector<Vector2T>;
using TVP3 = std::vector<Vector3T>;
using TVV = std::vector<Vector>;
using IVI = std::vector<Index>;
using IVI2 = std::vector<Vector2I>;
using IVI3 = std::vector<Vector3I>;
using IVI4 = std::vector<Vector4I>;

constexpr double pi = 3.14159265358979323846;

inline void TGSLAssert(const bool success, std::string flag) {
  if (!success) {
    throw std::runtime_error(flag);
  }
}

// std::vector<bool> does not work in C++!
// Google if you do not believe!
// Use BV = std::vector<unsigned char> instead for almost-as-good memory footprint
// but compatibility with STL containers, multithreading, etc.
// 1_uc == true
// 0_uc == false
inline constexpr unsigned char operator"" _uc(unsigned long long arg) noexcept {
  return static_cast<unsigned char>(arg);
}

// https://stackoverflow.com/a/20583932/4451284
template <typename T>
void remove_indices(std::vector<T>& vector, const std::vector<sz>& to_remove, const size_t element_size) {
  auto vector_base = vector.begin();
  typename std::vector<T>::size_type down_by = 0;

  for (auto iter = to_remove.cbegin(); iter < to_remove.cend(); iter++, down_by++) {
    typename std::vector<T>::size_type next_index_to_remove = (iter + 1 == to_remove.cend() ? vector.size() / element_size : *(iter + 1));

    std::move(vector_base + element_size * (*iter + 1), vector_base + element_size * next_index_to_remove, vector_base + element_size * (*iter - down_by));
  }
  vector.resize(vector.size() - element_size * to_remove.size());
}

inline void Log(int cfgLevel, int level, std::string msg) {
  if (level <= cfgLevel)
    std::cout << msg << std::endl;
}


#endif

