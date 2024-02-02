#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define NUM_IMAGES 3
#define KERNEL_SIZE 9
#define WEIGHT_SIZE NUM_IMAGES*KERNEL_SIZE*KERNEL_SIZE
__constant__ unsigned char WEIGHT_BYTES[WEIGHT_SIZE*sizeof(double)];
__constant__ unsigned char BIAS_BYTES[KERNEL_SIZE*sizeof(double)];

#define LOCATIONSPERBLOCK 16
#define NUM_THREADS_FORWARD 256
#define NUM_THREADS_INFERENCE 256

namespace {

template <typename scalar_t>
__global__ void sm_linear_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> image, // 3, N, N
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> y,
    const int nBlocksPerCopy, const int n1, const int n2) {

  __shared__ scalar_t I[NUM_IMAGES][LOCATIONSPERBLOCK+2][LOCATIONSPERBLOCK+2];

  const int location = blockIdx.x / nBlocksPerCopy;
  const int innerBlock = blockIdx.x % nBlocksPerCopy;

  const int N1 = n1 * LOCATIONSPERBLOCK;
  const int N2 = n2 * LOCATIONSPERBLOCK;

  const int j = (location % n2) * LOCATIONSPERBLOCK;
  const int i = (location / n2) * LOCATIONSPERBLOCK;

  int tid = threadIdx.x;
  const int width = LOCATIONSPERBLOCK+2;
  const int totalSize = NUM_IMAGES * width*width;
  while (tid < totalSize) {
    int ttid = tid;
    int iy = tid % width;
    tid /= width;
    int ix = tid % width;
    int m = tid / width;
    const int a = i+ix-1, b = j+iy-1;
    if (a >= 0 && a < N1 && b >= 0 && b < N2)
      I[m][ix][iy] = image[m][a][b];
    else
      I[m][ix][iy] = 0.0;
    tid = ttid + blockDim.x;
  }

  int p = innerBlock * blockDim.x + threadIdx.x;
  const int l = p % 3;
  p /= 3;
  const int k = p % 3;
  const int m = p / 3;

  if (m >= NUM_IMAGES) return;
  __syncthreads();

  scalar_t _y = 0.0;
  #pragma unroll(1)
  for (int _i = 0; _i < LOCATIONSPERBLOCK; ++_i) {
    for (int _j = 0; _j < LOCATIONSPERBLOCK; ++_j) {
      _y += I[m][_i+k][_j+l];
    }
  }

  atomicAdd(&y[0][m][k][l], _y);
}

} // namespace



std::vector<torch::Tensor> sm_linear_cuda_forward(
    torch::Tensor image,
    torch::Tensor weights,
    torch::Tensor bias) {

  assert(image.size(0) == NUM_IMAGES);

  const int N1 = image.size(1);
  const int N2 = image.size(2);

  const int nThreads = NUM_THREADS_FORWARD;
  const int nBlocksPerCopy = (NUM_IMAGES*KERNEL_SIZE + nThreads - 1) / nThreads;
  const int locationsPerBlock = LOCATIONSPERBLOCK;

  assert((N1 % locationsPerBlock == 0) && (N2 % locationsPerBlock == 0)); // Data must be divisible by divisions

  const int n1 = N1 / locationsPerBlock;
  const int n2 = N2 / locationsPerBlock;
  const dim3 threads(nThreads);
  const dim3 blocks(nBlocksPerCopy*n1*n2);

  auto y = torch::zeros({1, NUM_IMAGES, 3, 3}, torch::dtype(image.dtype()).device(image.device())); // useful for backward

  AT_DISPATCH_FLOATING_TYPES(image.type(), "sm_linear_forward_cuda", ([&] {
    sm_linear_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        y.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        nBlocksPerCopy, n1, n2);
  }));

  y /= N1*N2;
  auto y_sum = (weights.flatten(1).matmul(y.flatten())).mean() + bias.mean();
  return {y_sum, y};
}

std::vector<torch::Tensor> sm_linear_cuda_inference(
    torch::Tensor image,
    torch::Tensor weights,
    torch::Tensor bias) {

  assert(image.size(0) == NUM_IMAGES);

  const int N1 = image.size(1);
  const int N2 = image.size(2);

  const int nThreads = NUM_THREADS_INFERENCE;
  const int nBlocksPerCopy = (NUM_IMAGES*KERNEL_SIZE + nThreads - 1) / nThreads;
  const int locationsPerBlock = LOCATIONSPERBLOCK;

  assert((N1 % locationsPerBlock == 0) && (N2 % locationsPerBlock == 0)); // Data must be divisible by divisions

  const int n1 = N1 / locationsPerBlock;
  const int n2 = N2 / locationsPerBlock;
  const dim3 threads(nThreads);
  const dim3 blocks(nBlocksPerCopy*n1*n2);

  auto y = torch::zeros({1, NUM_IMAGES, 3, 3}, torch::dtype(image.dtype()).device(image.device())); // useful for backward

  AT_DISPATCH_FLOATING_TYPES(image.type(), "sm_linear_inference_cuda", ([&] {
    sm_linear_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        y.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        nBlocksPerCopy, n1, n2);
  }));
  y /= N1*N2;
  auto y_sum = (weights.flatten(1).matmul(y.flatten())).mean() + bias.mean();
  return {y_sum};
}

std::vector<torch::Tensor> sm_linear_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor y) {

  auto grad_b = torch::ones({9}, torch::dtype(grad_output.dtype()).device(grad_output.device())) / KERNEL_SIZE * grad_output;

  auto grad_w = y.expand({9, -1, -1, -1}) * grad_output / KERNEL_SIZE;

  return {grad_w, grad_b};
}
