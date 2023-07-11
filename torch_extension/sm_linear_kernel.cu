#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define KERNEL_SIZE 9
__constant__ unsigned char WEIGHT_BYTES[KERNEL_SIZE * KERNEL_SIZE * sizeof(double)];

#define NUM_THREADS_FORWARD 512
#define NUM_THREADS_BACKWARD 256

namespace {
template <typename scalar_t>
__global__ void sm_linear_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> image, // 1, N, N
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> y,
    const int nblocks) {

  __shared__ scalar_t z[NUM_THREADS_FORWARD];

  const int N1 = image.size(1) - 2;
  const int N2 = image.size(2) - 2;

  const scalar_t *WEIGHT = (const scalar_t *)(WEIGHT_BYTES);

  const int b = blockIdx.x / nblocks;
  const int innerBlock = blockIdx.x % nblocks;
  int location = blockDim.x * innerBlock + threadIdx.x;

  z[threadIdx.x] = 0.0;

  if (location < N1*N2) {
    const int j = location % N2;
    const int i = location / N2;

    for (int k = 0; k <= 2; ++k) {
      for (int l = 0; l <= 2; ++l) {
          z[threadIdx.x] += WEIGHT[9*b+3*k+l] * image[0][i+k][j+l];
      }
    }
  }
  __syncthreads();

  // reduction
  int data = blockDim.x;
  while (data > 1) {
    if (threadIdx.x < data / 2)
      z[threadIdx.x] += z[threadIdx.x + data / 2];
    data /= 2;
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(&y[0], z[0]);
}

template <typename scalar_t>
__global__ void sm_linear_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> image, // 1, N, N
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_w,
    const int locationsPerBlock) {

  __shared__ scalar_t d_w[KERNEL_SIZE*KERNEL_SIZE];

  const int N1 = (image.size(1) - 2) / locationsPerBlock;
  const int N2 = (image.size(2) - 2) / locationsPerBlock;

  const int nBlocksPerCopy = (KERNEL_SIZE*KERNEL_SIZE + blockDim.x - 1) / blockDim.x;

  const int block = blockIdx.x / nBlocksPerCopy;
  const int innerBlock = blockIdx.x % nBlocksPerCopy;

  int location = block;
  const int j = (location % N2) * locationsPerBlock;
  const int i = (location / N2) * locationsPerBlock;

  const int p = innerBlock * blockDim.x + threadIdx.x;

  if (p >= KERNEL_SIZE*KERNEL_SIZE) return; // Wasted threads

  const int pp = p % KERNEL_SIZE;
  const int l = pp % 3;
  const int k = pp / 3;


  const int p0 = p / KERNEL_SIZE;

  const int c = 9*p0+3*k+l;
  d_w[c] = 0.0;

  for (int _i = 0; _i < locationsPerBlock; ++_i) {
    for (int _j = 0; _j < locationsPerBlock; ++_j) {
      d_w[c] += image[0][i+_i+k][j+_j+l];
    }
  }

  atomicAdd(&grad_w[p0][0][k][l], d_w[c]);
}

} // namespace

std::vector<torch::Tensor> sm_linear_cuda_forward(
    torch::Tensor image,
    torch::Tensor weights,
    torch::Tensor bias) {

  if (image.dtype() == torch::ScalarType::Double) {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<double>(), KERNEL_SIZE*KERNEL_SIZE * sizeof(double));
  } else {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<float>(), KERNEL_SIZE*KERNEL_SIZE * sizeof(float));
  }

  const int N = image.size(2)-2;
  auto y = torch::zeros({1}, torch::dtype(image.dtype()).device(image.device()));

  const int nthreads = NUM_THREADS_FORWARD;
  const int nblocks = (N*N + nthreads - 1) / nthreads; // b, i, j

  const dim3 threads(nthreads);
  const dim3 blocks(nblocks * KERNEL_SIZE);

  AT_DISPATCH_FLOATING_TYPES(image.type(), "sm_linear_forward_cuda", ([&] {
    sm_linear_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        y.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        nblocks);
  }));

  y /= KERNEL_SIZE * std::pow(N, 2);
  y += bias.mean();
  return {y};
}

std::vector<torch::Tensor> sm_linear_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor image) {

  const int N = image.size(2) - 2;
  const int totalData = N * N;

  const int nThreads = NUM_THREADS_BACKWARD;

  const int nBlocksPerCopy = (KERNEL_SIZE*KERNEL_SIZE + nThreads - 1) / nThreads;

  const int locationsPerBlock = 4;

  assert(N % locationsPerBlock == 0); // Data must be divisible by divisions

  const dim3 threads(nThreads);
  const dim3 blocks(nBlocksPerCopy*(totalData/std::pow(locationsPerBlock, 2)));

  auto grad_w = torch::zeros({9, 1, 3, 3}, torch::dtype(image.dtype()).device(image.device()));
  auto grad_b = torch::ones({9}, torch::dtype(image.dtype()).device(image.device())) / KERNEL_SIZE * grad_output;

  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "sm_linear_cuda_backward", ([&] {
    sm_linear_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_w.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        locationsPerBlock);
  }));
  grad_w /= KERNEL_SIZE * std::pow(N, 2);
  grad_w *= grad_output;
  return {grad_w, grad_b};
}
