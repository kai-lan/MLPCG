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
#define NUM_THREADS_FORWARD 512
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
    I[m][ix][iy] = image[m][i+ix][j+iy];
    tid = ttid + blockDim.x;
  }

  int p = innerBlock * blockDim.x + threadIdx.x;
  const int l = p % 3;
  p /= 3;
  const int k = p % 3;
  p /= 3;
  const int m = p % NUM_IMAGES;
  const int c = p / NUM_IMAGES;

  if (c >= KERNEL_SIZE) return; // Wasted threads
  __syncthreads();

  scalar_t _y = 0.0;
  #pragma unroll(1)
  for (int _i = 0; _i < LOCATIONSPERBLOCK; ++_i) {
    for (int _j = 0; _j < LOCATIONSPERBLOCK; ++_j) {
      _y += I[m][_i+k][_j+l];
    }
  }

  atomicAdd(&y[c][m][k][l], _y);
}

// template <size_t blockSize, typename T>
// __device__ void warpReduce(volatile T *sdata, size_t tid)
// {
//     if (blockSize >= 64) sdata[tid] += sdata[tid + 32]; __syncthreads();
//     if (blockSize >= 32) sdata[tid] += sdata[tid + 16]; __syncthreads();
//     if (blockSize >= 16) sdata[tid] += sdata[tid +  8]; __syncthreads();
//     if (blockSize >=  8) sdata[tid] += sdata[tid +  4]; __syncthreads();
//     if (blockSize >=  4) sdata[tid] += sdata[tid +  2]; __syncthreads();
//     if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
// }

// template <typename scalar_t>
// __global__ void sm_linear_cuda_inference_kernel(
//     const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> image, // 3, N, N
//     torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> y,
//     const int nBlocks, const int N1, const int N2) {

//   __shared__ scalar_t z[NUM_THREADS_INFERENCE];

//   const scalar_t *WEIGHT = (const scalar_t *)(WEIGHT_BYTES);

//   const int outerBlock = blockIdx.x / nBlocks;
//   const int innerBlock = blockIdx.x % nBlocks;
//   const int c = outerBlock;
//   const int tid = threadIdx.x;
//   const int location = blockDim.x * innerBlock + tid;

//   z[tid] = 0.0;

//   if (location < NUM_IMAGES*N1*N2) {
//     const int j = location % N2;
//     const int i = location / N2;

//     int c0, c1;
//     #pragma unroll(1)
//     for (int m = 0; m < NUM_IMAGES; ++m) {
//       c0 = NUM_IMAGES*c+m;
//       for (int k = 0; k <= 2; ++k) {
//         c1 = 3*c0+k;
//         z[tid] += WEIGHT[3*c1] * image[m][i+k][j];
//         z[tid] += WEIGHT[3*c1+1] * image[m][i+k][j+1];
//         z[tid] += WEIGHT[3*c1+2] * image[m][i+k][j+2];
//       }
//     }
//   }
//   __syncthreads();

//   // reduction
//   if (NUM_THREADS_INFERENCE >= 1024) { if (tid < 512) { z[tid] += z[tid + 512]; } __syncthreads(); }
//   if (NUM_THREADS_INFERENCE >=  512) { if (tid < 256) { z[tid] += z[tid + 256]; } __syncthreads(); }
//   if (NUM_THREADS_INFERENCE >=  256) { if (tid < 128) { z[tid] += z[tid + 128]; } __syncthreads(); }
//   if (NUM_THREADS_INFERENCE >=  128) { if (tid <  64) { z[tid] += z[tid +  64]; } __syncthreads(); }

//   if (tid < 32) warpReduce<NUM_THREADS_INFERENCE, scalar_t>(z, tid);
//   if (tid == 0) atomicAdd(&y[0], z[0]);
// }

template <typename scalar_t>
__global__ void sm_linear_cuda_inference_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> image, // num_imgs, N, N
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> y,
    const int N1, const int N2) { // bs, 1, N, N

  const scalar_t *WEIGHT = (const scalar_t *)(WEIGHT_BYTES);
  const scalar_t *BIAS = (const scalar_t *)(BIAS_BYTES);

  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadId >= N1*N2) return;
  const int j = threadId % N2;
  const int i = threadId / N2;

  scalar_t zz = 0.0;
  #pragma unroll
  for (int k = 0; k <= 2; ++k) {
    for (int l = 0; l <= 2; ++l) {
      int p = 3 * k + l;
      zz += BIAS[p];
      #pragma unroll
      for (int s = 0; s < NUM_IMAGES; ++s) {
        for (int m = 0; m <= 2; ++m) {
          for (int n = 0; n <= 2; ++n) {
            zz += WEIGHT[3*(3*(NUM_IMAGES*p+s)+m)+n] * image[s][i+m][j+n];
          }
        }
      }
    }
  }
  y[i][j] = zz;
}

} // namespace



std::vector<torch::Tensor> sm_linear_cuda_forward(
    torch::Tensor image,
    torch::Tensor weights,
    torch::Tensor bias) {

  assert(image.size(0) == NUM_IMAGES);

  const int N1 = image.size(1)-2;
  const int N2 = image.size(2)-2;

  const int nThreads = NUM_THREADS_FORWARD;
  const int nBlocksPerCopy = (WEIGHT_SIZE + nThreads - 1) / nThreads;
  const int locationsPerBlock = LOCATIONSPERBLOCK;

  assert((N1 % locationsPerBlock == 0) && (N2 % locationsPerBlock == 0)); // Data must be divisible by divisions

  const int n1 = N1 / locationsPerBlock;
  const int n2 = N2 / locationsPerBlock;
  const dim3 threads(nThreads);
  const dim3 blocks(nBlocksPerCopy*n1*n2);

  auto y = torch::zeros({9, NUM_IMAGES, 3, 3}, torch::dtype(image.dtype()).device(image.device())); // useful for backward
  auto y_sum = torch::zeros({1}, torch::dtype(image.dtype()).device(image.device()));

  AT_DISPATCH_FLOATING_TYPES(image.type(), "sm_linear_forward_cuda", ([&] {
    sm_linear_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        y.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        nBlocksPerCopy, n1, n2);
  }));

  y /= KERNEL_SIZE * N1*N2;
  y_sum[0] = weights.ravel().dot(y.ravel());
  y_sum += bias.mean();
  return {y_sum, y};
}

// std::vector<torch::Tensor> sm_linear_cuda_inference(
//     torch::Tensor image,
//     torch::Tensor weights,
//     torch::Tensor bias) {

//   assert(image.size(0) == NUM_IMAGES);

//   if (image.dtype() == torch::ScalarType::Double) {
//     cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<double>(), WEIGHT_SIZE * sizeof(double));
//   } else {
//     cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<float>(), WEIGHT_SIZE * sizeof(float));
//   }

//   const int N1 = image.size(1)-2;
//   const int N2 = image.size(2)-2;
//   auto y = torch::zeros({1}, torch::dtype(image.dtype()).device(image.device()));

//   const int nThreads = NUM_THREADS_INFERENCE;
//   const int nBlocks = (N1*N2 + nThreads - 1) / nThreads; // c, i, j

//   const dim3 threads(nThreads);
//   const dim3 blocks(nBlocks * KERNEL_SIZE);

//   AT_DISPATCH_FLOATING_TYPES(image.type(), "sm_linear_forward_cuda", ([&] {
//     sm_linear_cuda_inference_kernel<scalar_t><<<blocks, threads>>>(
//         image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//         y.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
//         nBlocks, N1, N2);
//   }));

//   y /= KERNEL_SIZE * N1 * N2;
//   y += bias.mean();
//   return {y};
// }

std::vector<torch::Tensor> sm_linear_cuda_inference(
    torch::Tensor image,
    torch::Tensor weights,
    torch::Tensor bias) {

  assert(image.size(0) == NUM_IMAGES);

  if (image.dtype() == torch::ScalarType::Double) {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<double>(), WEIGHT_SIZE * sizeof(double));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<double>(), KERNEL_SIZE * sizeof(double));
  } else {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<float>(), WEIGHT_SIZE * sizeof(float));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<float>(), KERNEL_SIZE * sizeof(float));
  }

  const int N1 = image.size(1)-2;
  const int N2 = image.size(2)-2;
  auto y = torch::zeros({N1, N2}, torch::dtype(image.dtype()).device(image.device()));

  const int nThreads = NUM_THREADS_INFERENCE;
  const int nBlocks = (N1*N2 + nThreads - 1) / nThreads;

  const dim3 threads(nThreads);
  const dim3 blocks(nBlocks);

  AT_DISPATCH_FLOATING_TYPES(image.type(), "sm_linear_inference_cuda", ([&] {
    sm_linear_cuda_inference_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        N1, N2);
  }));
  auto y_sum = y.mean() / KERNEL_SIZE;
  return {y_sum};
}

std::vector<torch::Tensor> sm_linear_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor y) {

  auto grad_b = torch::ones({9}, torch::dtype(grad_output.dtype()).device(grad_output.device())) / KERNEL_SIZE * grad_output;

  auto grad_w = y * grad_output;

  return {grad_w, grad_b};
}
