#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/cuda/Atomic.cuh>
#include <vector>

#define NUM_IMAGES 3
#define KERNEL_SIZE 27
#define WEIGHT_SIZE NUM_IMAGES*KERNEL_SIZE*KERNEL_SIZE

#define LOCATIONSPERBLOCK 8
#define NUM_THREADS_INFERENCE 256
#define NUM_THREADS_FORWARD 256
#define NUM_THREADS_BACKWARD 256

namespace {

template <typename scalar_t>
__global__ void sm_linear_3d_cuda_inference_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image, // 3, N, N, N
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> y,
    const int nBlocksPerCopy, const int n1, const int n2, const int n3,
    const int y_numel) {

  __shared__ scalar_t I[NUM_IMAGES][LOCATIONSPERBLOCK+2][LOCATIONSPERBLOCK+2][LOCATIONSPERBLOCK+2];

  const int location = blockIdx.x / nBlocksPerCopy;
  const int innerBlock = blockIdx.x % nBlocksPerCopy;

  const int N1 = n1 * LOCATIONSPERBLOCK;
  const int N2 = n2 * LOCATIONSPERBLOCK;
  const int N3 = n3 * LOCATIONSPERBLOCK;

  const int ij = (location % n3) * LOCATIONSPERBLOCK;
  const int j = ((location/n3) % n2) * LOCATIONSPERBLOCK;
  const int i = ((location/n3) / n2) * LOCATIONSPERBLOCK;

  int tid = threadIdx.x;
  const int width = LOCATIONSPERBLOCK+2;
  const int totalSize = NUM_IMAGES * width*width*width;
  while (tid < totalSize) {
    int ttid = tid;
    int iz = tid % width;
    tid /= width;
    int iy = tid % width;
    tid /= width;
    int ix = tid % width;
    int m = tid / width;
    const int a = i+ix-1, b = j+iy-1, c = ij+iz-1;
    if (a >= 0 && a < N1 && b >= 0 && b < N2 && c >= 0 && c < N3)
      I[m][ix][iy][iz] = image[m][a][b][c];
    else
      I[m][ix][iy][iz] = 0.0;
    tid = ttid + blockDim.x;
  }

  int p = innerBlock * blockDim.x + threadIdx.x;
  const int kl = p % 3;
  p /= 3;
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
      for (int _ij = 0; _ij < LOCATIONSPERBLOCK; ++_ij) {
        _y += I[m][_i+k][_j+l][_ij+kl];
      }
    }
  }
  const int index = m * y.stride(1) + k * y.stride(2) + l * y.stride(3) + kl * y.stride(4);
  at::native::fastAtomicAdd(y.data(), index, y_numel, _y, true);
}

template <typename scalar_t>
__global__ void sm_linear_3d_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image, // 3, N, N, N
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> y,
    const int nBlocksPerCopy, const int n1, const int n2, const int n3,
    const int y_numel) {

  __shared__ scalar_t I[NUM_IMAGES][LOCATIONSPERBLOCK+2][LOCATIONSPERBLOCK+2][LOCATIONSPERBLOCK+2];

  const int location = blockIdx.x / nBlocksPerCopy;
  const int innerBlock = blockIdx.x % nBlocksPerCopy;

  const int N1 = n1 * LOCATIONSPERBLOCK;
  const int N2 = n2 * LOCATIONSPERBLOCK;
  const int N3 = n3 * LOCATIONSPERBLOCK;

  const int ij = (location % n3) * LOCATIONSPERBLOCK;
  const int j = ((location/n3) % n2) * LOCATIONSPERBLOCK;
  const int i = ((location/n3) / n2) * LOCATIONSPERBLOCK;

  int tid = threadIdx.x;
  const int width = LOCATIONSPERBLOCK+2;
  const int totalSize = NUM_IMAGES * width*width*width;
  while (tid < totalSize) {
    int ttid = tid;
    int iz = tid % width;
    tid /= width;
    int iy = tid % width;
    tid /= width;
    int ix = tid % width;
    int m = tid / width;
    const int a = i+ix-1, b = j+iy-1, c = ij+iz-1;
    if (a >= 0 && a < N1 && b >= 0 && b < N2 && c >= 0 && c < N3)
      I[m][ix][iy][iz] = image[m][a][b][c];
    else
      I[m][ix][iy][iz] = 0.0;
    tid = ttid + blockDim.x;
  }

  int p = innerBlock * blockDim.x + threadIdx.x;
  const int kl = p % 3;
  p /= 3;
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
      for (int _ij = 0; _ij < LOCATIONSPERBLOCK; ++_ij) {
        _y += I[m][_i+k][_j+l][_ij+kl];
      }
    }
  }
  const int index = m * y.stride(1) + k * y.stride(2) + l * y.stride(3) + kl * y.stride(4);
  at::native::fastAtomicAdd(y.data(), index, y_numel, _y, true);
}

} // namespace

std::vector<torch::Tensor> sm_linear_3d_cuda_inference(
    torch::Tensor image,
    torch::Tensor weights,
    torch::Tensor bias) {

  assert(image.size(0) == NUM_IMAGES);

  const int N1 = image.size(1);
  const int N2 = image.size(2);
  const int N3 = image.size(3);

  const int nThreads = NUM_THREADS_INFERENCE;
  const int nBlocksPerCopy = (NUM_IMAGES*KERNEL_SIZE + nThreads - 1) / nThreads;
  const int locationsPerBlock = LOCATIONSPERBLOCK;

  assert((N1 % locationsPerBlock == 0) && (N2 % locationsPerBlock == 0) && (N3 % locationsPerBlock == 0)); // Data must be divisible by divisions

  const int n1 = N1 / locationsPerBlock;
  const int n2 = N2 / locationsPerBlock;
  const int n3 = N3 / locationsPerBlock;
  const dim3 threads(nThreads);
  const dim3 blocks(nBlocksPerCopy*n1*n2*n3);

  auto y = torch::zeros({1, NUM_IMAGES, 3, 3, 3}, torch::dtype(image.dtype()).device(image.device())); // useful for backward

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "sm_linear_3d_inference_cuda", ([&] {
    sm_linear_3d_cuda_inference_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        y.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        nBlocksPerCopy, n1, n2, n3,
        image.numel());
  }));
  y /= N1*N2*N3;
  auto y_sum = (weights.flatten(1).matmul(y.flatten())).mean() + bias.mean();
  return {y_sum, y};
}

std::vector<torch::Tensor> sm_linear_3d_cuda_forward(
    torch::Tensor image,
    torch::Tensor weights,
    torch::Tensor bias) {

  assert(image.size(0) == NUM_IMAGES);

  const int N1 = image.size(1);
  const int N2 = image.size(2);
  const int N3 = image.size(3);

  const int nThreads = NUM_THREADS_FORWARD;
  const int nBlocksPerCopy = (NUM_IMAGES*KERNEL_SIZE + nThreads - 1) / nThreads;
  const int locationsPerBlock = LOCATIONSPERBLOCK;

  assert((N1 % locationsPerBlock == 0) && (N2 % locationsPerBlock == 0) && (N3 % locationsPerBlock == 0)); // Data must be divisible by divisions

  const int n1 = N1 / locationsPerBlock;
  const int n2 = N2 / locationsPerBlock;
  const int n3 = N3 / locationsPerBlock;
  const dim3 threads(nThreads);
  const dim3 blocks(nBlocksPerCopy*n1*n2*n3);

  auto y = torch::zeros({1, NUM_IMAGES, 3, 3, 3}, torch::dtype(image.dtype()).device(image.device())); // useful for backward

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "sm_linear_3d_forward_cuda", ([&] {
    sm_linear_3d_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        y.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        nBlocksPerCopy, n1, n2, n3,
        image.numel());
  }));
  y /= N1*N2*N3;
  auto y_sum = (weights.flatten(1).matmul(y.flatten())).mean() + bias.mean();
  return {y_sum, y};
}

std::vector<torch::Tensor> sm_linear_3d_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor y) { // computed from forward

  auto grad_b = torch::ones({27}, torch::dtype(grad_output.dtype()).device(grad_output.device())) / KERNEL_SIZE * grad_output;

  auto grad_w = y.expand({27, -1, -1, -1, -1}) * grad_output / KERNEL_SIZE;

  return {grad_w, grad_b};
}

int main() {
  int bs = 5;
  int N = 128;
  int num_imgs = 3;
  auto image = torch::ones({num_imgs, N, N, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto padded_image = torch::ones({num_imgs, N+2, N+2, N+2}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto x = torch::rand({bs, 1, N, N, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto weight = torch::rand({27, num_imgs, 3, 3, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto bias = torch::rand({27}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto grad_output = torch::ones({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto y = torch::zeros({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  // curandState *d_states;
  // cudaMalloc(&d_states, CNT * sizeof(curandState));
  // kernel_setup_randstates_2d<<<1,CNT>>>(d_states, 1,1, 1);
  // cudaDeviceSynchronize();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < 100; ++i)
    // y = sm_linear_3d_cuda_backward(grad_output, weight)[0];
    y = sm_linear_3d_cuda_forward(padded_image, weight, bias)[0];

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "time " << milliseconds/100*1000 << " us" << std::endl;
  // cudaDeviceSynchronize();
}