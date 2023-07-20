#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define NUM_IMAGES 3
#define KERNEL_SIZE 9
#define WEIGHT_SIZE NUM_IMAGES*KERNEL_SIZE*KERNEL_SIZE
__constant__ unsigned char WEIGHT_BYTES[WEIGHT_SIZE*sizeof(double)];

#define LOCATIONSPERBLOCK 4
#define NUM_THREADS_FORWARD 512
#define NUM_THREADS_BACKWARD 256

namespace {
template <typename scalar_t>
__global__ void sm_linear_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> image, // 3, N, N
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> y,
    const int nBlocks, const int N1, const int N2) {

  __shared__ scalar_t z[NUM_THREADS_FORWARD];

  const scalar_t *WEIGHT = (const scalar_t *)(WEIGHT_BYTES);

  const int c = blockIdx.x / nBlocks;
  const int innerBlock = blockIdx.x % nBlocks;
  const int location = blockDim.x * innerBlock + threadIdx.x;

  z[threadIdx.x] = 0.0;

  if (location < N1*N2) {
    const int j = location % N2;
    const int i = location / N2;

    #pragma unroll
    for (int m = 0; m < NUM_IMAGES; ++m) {
      for (int k = 0; k <= 2; ++k) {
        for (int l = 0; l <= 2; ++l) {
            z[threadIdx.x] += WEIGHT[3*(3*(NUM_IMAGES*c+m)+k)+l] * image[m][i+k][j+l];
        }
      }
    }
  }
  __syncthreads();

  // reduction
  #pragma unroll
  for (int n = NUM_THREADS_FORWARD; n > 1; n /= 2) {
    if (threadIdx.x < n / 2)
      z[threadIdx.x] += z[threadIdx.x + n / 2];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(&y[0], z[0]);
}

template <typename scalar_t>
__global__ void sm_linear_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> image, // 1, N, N
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_w,
    const int nBlocksPerCopy, const int n1, const int n2) {

  __shared__ scalar_t d_w[WEIGHT_SIZE];

  const int location = blockIdx.x / nBlocksPerCopy;
  const int innerBlock = blockIdx.x % nBlocksPerCopy;

  const int j = (location % n2) * LOCATIONSPERBLOCK;
  const int i = (location / n2) * LOCATIONSPERBLOCK;

  int p = innerBlock * blockDim.x + threadIdx.x;

  if (p >= WEIGHT_SIZE) return; // Wasted threads

  const int idx = p;
  const int l = p % 3;
  p /= 3;
  const int k = p % 3;
  p /= 3;
  const int m = p % NUM_IMAGES;
  const int c = p / NUM_IMAGES;

  d_w[idx] = 0.0;
  #pragma unroll
  for (int _i = 0; _i < LOCATIONSPERBLOCK; ++_i) {
    for (int _j = 0; _j < LOCATIONSPERBLOCK; ++_j) {
      d_w[idx] += image[m][i+_i+k][j+_j+l];
    }
  }
  atomicAdd(&grad_w[c][m][k][l], d_w[idx]);
}
} // namespace



std::vector<torch::Tensor> sm_linear_cuda_forward(
    torch::Tensor image,
    torch::Tensor weights,
    torch::Tensor bias) {

  assert(image.size(0) == NUM_IMAGES);

  if (image.dtype() == torch::ScalarType::Double) {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<double>(), WEIGHT_SIZE * sizeof(double));
  } else {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<float>(), WEIGHT_SIZE * sizeof(float));
  }

  const int N1 = image.size(1)-2;
  const int N2 = image.size(2)-2;
  auto y = torch::zeros({1}, torch::dtype(image.dtype()).device(image.device()));

  const int nThreads = NUM_THREADS_FORWARD;
  const int nBlocks = (N1*N2 + nThreads - 1) / nThreads; // c, i, j

  const dim3 threads(nThreads);
  const dim3 blocks(nBlocks * KERNEL_SIZE);

  AT_DISPATCH_FLOATING_TYPES(image.type(), "sm_linear_forward_cuda", ([&] {
    sm_linear_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        y.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        nBlocks, N1, N2);
  }));

  y /= KERNEL_SIZE * N1 * N2;
  y += bias.mean();
  return {y};
}

std::vector<torch::Tensor> sm_linear_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor image) {

  assert(image.size(0) == NUM_IMAGES);

  const int N1 = image.size(1) - 2;
  const int N2 = image.size(2) - 2;

  const int nThreads = NUM_THREADS_BACKWARD;
  const int nBlocksPerCopy = (WEIGHT_SIZE + nThreads - 1) / nThreads;
  const int locationsPerBlock = LOCATIONSPERBLOCK;

  assert((N1 % locationsPerBlock == 0) && (N2 % locationsPerBlock == 0)); // Data must be divisible by divisions

  const int n1 = N1 / locationsPerBlock;
  const int n2 = N2 / locationsPerBlock;
  const dim3 threads(nThreads);
  const dim3 blocks(nBlocksPerCopy*n1*n2);

  auto grad_w = torch::zeros({9, NUM_IMAGES, 3, 3}, torch::dtype(image.dtype()).device(image.device()));
  auto grad_b = torch::ones({9}, torch::dtype(image.dtype()).device(image.device())) / KERNEL_SIZE * grad_output;

  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "sm_linear_cuda_backward", ([&] {
    sm_linear_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_w.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        nBlocksPerCopy, n1, n2);
  }));
  grad_w /= KERNEL_SIZE * N1 * N2;
  grad_w *= grad_output;
  return {grad_w, grad_b};
}
