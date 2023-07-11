#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define KERNEL_SIZE 9
__constant__ unsigned char WEIGHT_BYTES[KERNEL_SIZE * KERNEL_SIZE * sizeof(double)];
__constant__ unsigned char BIAS_BYTES[KERNEL_SIZE * sizeof(double)];

#define LOCATIONS_PER_BLOCK 8
#define NUM_THREADS_FORWARD 512
#define NUM_THREADS_BACKWARD 128

namespace {
template <typename scalar_t>
__global__ void sm_block_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> image, // 1, N, N
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> x, // bs, 1, N, N
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> y,
    const int B,
    const int N1, const int N2) { // bs, 1, N, N

  const scalar_t *WEIGHT = (const scalar_t *)(WEIGHT_BYTES);
  const scalar_t *BIAS = (const scalar_t *)(BIAS_BYTES);

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = threadId % N2;
  threadId /= N2;
  const int i = threadId % N1;
  const int b = threadId / N1;

  if (b < 0 || b >= B) return;

  scalar_t z = 0.0;
  #pragma unroll
  for (int k = 0; k <= 2; ++k) {
    for (int l = 0; l <= 2; ++l) {
      int p = 3 * k + l;
      z = BIAS[p];
      #pragma unroll
      for (int m = 0; m <= 2; ++m) {
        for (int n = 0; n <= 2; ++n) {
          z += WEIGHT[9*p+3*m+n] * image[0][i+m][j+n];
        }
      }
      z *= x[b][0][i+k][j+l];
      y[b][0][i][j] += z;
    }
  }
} // forward kernel

template <typename scalar_t>
__global__ void sm_block_cuda_dwdb_fast_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_output, // bs, 1, N, N
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> image, // 1, N, N
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> x, // bs, 1, N, N
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_w,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> grad_b,
    const int nBlocksPerCopy,
    const int B,
    const int N1, const int N2) {

  __shared__ scalar_t d_w[KERNEL_SIZE*KERNEL_SIZE];
  __shared__ scalar_t d_b[KERNEL_SIZE];

  const int n1 = N1 / LOCATIONS_PER_BLOCK;
  const int n2 = N2 / LOCATIONS_PER_BLOCK;

  const int block = blockIdx.x / nBlocksPerCopy;
  const int innerBlock = blockIdx.x % nBlocksPerCopy;

  int location = block;
  const int j = (location % n2) * LOCATIONS_PER_BLOCK;
  location /= n2;
  const int i = (location % n1) * LOCATIONS_PER_BLOCK;
  const int b = location / n1;


  const int p = innerBlock * blockDim.x + threadIdx.x;

  if (p >= (KERNEL_SIZE+1)*KERNEL_SIZE) return; // Wasted threads

  const int pp = p % KERNEL_SIZE;
  const int l = pp % 3;
  const int k = pp / 3;

  const int p0 = p / KERNEL_SIZE;

  int n, m;
  int c;

  if (p0 == KERNEL_SIZE) {
    d_b[pp] = 0.0;
    #pragma unroll
    for (int _i = 0; _i < LOCATIONS_PER_BLOCK; ++_i) {
      for (int _j = 0; _j < LOCATIONS_PER_BLOCK; ++_j) {
        d_b[pp] += grad_output[b][0][i+_i][j+_j] * x[b][0][i+_i+k][j+_j+l];
      }
    }
  } else {
    n = p0 % 3;
    m = p0 / 3;
    c = 9*pp + 3*m + n;
    d_w[c] = 0.0;
    #pragma unroll
    for (int _i = 0; _i < LOCATIONS_PER_BLOCK; ++_i) {
      for (int _j = 0; _j < LOCATIONS_PER_BLOCK; ++_j) {
        d_w[c] += grad_output[b][0][i+_i][j+_j] * image[0][i+_i+m][j+_j+n] * x[b][0][i+_i+k][j+_j+l];
      }
    }
  }

  if (p0 == KERNEL_SIZE)
    atomicAdd(&grad_b[pp], d_b[pp]);
  else
    atomicAdd(&grad_w[pp][0][m][n], d_w[c]);
} // backward dw, db

template <typename scalar_t>
__global__ void sm_block_cuda_dx_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_output, // bs, 1, N, N
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> image, // 1, N, N
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> x, // bs, 1, N, N
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_x,
    const int B,
    const int N1, const int N2) {

  const scalar_t *WEIGHT = (const scalar_t *)(WEIGHT_BYTES);
  const scalar_t *BIAS = (const scalar_t *)(BIAS_BYTES);

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId < B * N1 * N2) {
    const int j = threadId % N2;

    threadId /= N2;
    const int i = threadId % N1;

    threadId /= N1;
    const int b = threadId % B;

    scalar_t z = 0.0;
    for (int k = 0; k <= 2; ++k) {
      for (int l = 0; l <= 2; ++l) {
        int q = 3 * k + l;

        int ii = i + 1 - k, jj = j + 1 - l;
        if (ii < 0 || ii >= N1 || jj < 0 || jj >= N2) continue;

        z = BIAS[q];
        for (int m = 0; m <= 2; ++m) {
          for (int n = 0; n <= 2; ++n) {
            z += WEIGHT[9*q+3*m+n] * image[0][ii+m][jj+n];
          }
        }
        z *= grad_output[b][0][ii][jj];
        grad_x[b][0][i][j] += z;

      }
    }
  }
} // backward dx

} // namespace

std::vector<torch::Tensor> sm_block_cuda_forward(
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {

  if (x.dtype() == torch::ScalarType::Double) {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<double>(), KERNEL_SIZE*KERNEL_SIZE * sizeof(double));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<double>(), KERNEL_SIZE * sizeof(double));
  } else {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<float>(), KERNEL_SIZE*KERNEL_SIZE * sizeof(float));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<float>(), KERNEL_SIZE * sizeof(float));
  }

  const int B = x.size(0);
  const int N1 = image.size(1)-2;
  const int N2 = image.size(2)-2;
  auto y = torch::zeros({B, x.size(1), N1, N2}, torch::dtype(x.dtype()).device(x.device()));

  const int nthreads = NUM_THREADS_FORWARD; // for 2D num_threads = 32*32 = 1024
  const int nblocks = (B*N1*N2 + nthreads - 1) / nthreads;

  const dim3 threads(nthreads);
  const dim3 blocks(nblocks);

  AT_DISPATCH_FLOATING_TYPES(x.type(), "sm_block_forward_cuda", ([&] {
    sm_block_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        y.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        B,
        N1, N2);
  }));

  return {y};
}

std::vector<torch::Tensor> sm_block_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {

  if (x.dtype() == torch::ScalarType::Double) {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<double>(), KERNEL_SIZE*KERNEL_SIZE * sizeof(double));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<double>(), KERNEL_SIZE * sizeof(double));
  } else {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<float>(), KERNEL_SIZE*KERNEL_SIZE * sizeof(float));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<float>(), KERNEL_SIZE * sizeof(float));
  }

  const int B = x.size(0);
  const int N1 = image.size(1)-2;
  const int N2 = image.size(2)-2;
  const int totalData = B * N1 * N2;

  const int nThreads = NUM_THREADS_BACKWARD;
  const int nBlocksPerCopy = ((KERNEL_SIZE+1)*KERNEL_SIZE + nThreads - 1) / nThreads;
  const int locationsPerBlock = LOCATIONS_PER_BLOCK;

  assert(N1 % locationsPerBlock == 0); // Data must be divisible by divisions
  assert(N2 % locationsPerBlock == 0); // Data must be divisible by divisions

  const dim3 threads(nThreads);
  const dim3 blocks(nBlocksPerCopy*(totalData/ std::pow(locationsPerBlock, 2))), bblocks((totalData + nThreads - 1) / nThreads);

  auto grad_x = torch::zeros({B, x.size(1), N1, N2}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_w = torch::zeros({9, 1, 3, 3}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_b = torch::zeros({9}, torch::dtype(x.dtype()).device(x.device()));

  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "sm_block_cuda_dwdb", ([&] {
    sm_block_cuda_dwdb_fast_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_w.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_b.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        nBlocksPerCopy,
        B,
        N1, N2);
  }));
  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "sm_block_cuda_dx", ([&] {
    sm_block_cuda_dx_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        B,
        N1, N2);
  }));

  return {grad_x, grad_w, grad_b};
}