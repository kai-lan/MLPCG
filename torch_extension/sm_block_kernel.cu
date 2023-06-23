#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
template <typename scalar_t>
__global__ void sm_block_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> image, // 1, N, N
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> x, // bs, 1, N, N
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weights, // 9, 1, 3, 3
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> bias, // 9
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> y) { // bs, 1, N, N

  const int B = x.size(0);
  const int N1 = image.size(1) - 2;
  const int N2 = image.size(2) - 2;

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = threadId % N2;
  threadId /= N2;
  const int i = threadId % N1;
  const int b = threadId / N1;

  if (b < 0 || b >= B) return;
  if (i < 0 || i >= N1) return;
  if (j < 0 || j >= N2) return;

  // for (int b = 0; b < bs; ++b) {
    // for (int i = 1; i <= N; ++i) {
    //   for (int j = 1; j <= N; ++j) {
  scalar_t z = 0.0;
  for (int k = 0; k <= 2; ++k) {
    for (int l = 0; l <= 2; ++l) {
      int p = 3 * k + l;
      z = bias[p];
      for (int m = 0; m <= 2; ++m) {
        for (int n = 0; n <= 2; ++n) {
          // K[m][n] += weights[p][0][k+1][l+1] * image[0][i+k][j+l] + bias[p];
          z += weights[p][0][m][n] * image[0][i+m][j+n];
        }
      }
      z *= x[b][0][i+k][j+l];
      y[b][0][i][j] += z;
      // y[b][0][i][j] += K[m][n] * x[b][0][i+m-1][j+n-1];
    }
  }
    //   }
    // }
  // }

}

template <typename scalar_t>
__global__ void sm_block_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> image,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weights,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> bias,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_x,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_w,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> grad_b) {

  const int B = x.size(0);
  const int N1 = image.size(1) - 2;
  const int N2 = image.size(2) - 2;

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = threadId % N2;
  threadId /= N2;
  const int i = threadId % N1;
  const int b = threadId / N1;

  if (b < 0 || b >= B) return;
  if (i < 0 || i >= N1) return;
  if (j < 0 || j >= N2) return;

  // for (int b = 0; b < bs; ++b) {
    // for (int i = 0; i < N; ++i) {
    //   for (int j = 0; j < N; ++j) {
        for (int k = 0; k <= 2; ++k) {
          for (int l = 0; l <= 2; ++l) {
            int p = 3 * k + l;
            for (int m = 0; m <= 2; ++m) {
              for (int n = 0; n <= 2; ++n) {
                atomicAdd(&grad_w[p][0][m][n], grad_output[b][0][i][j] * image[0][i+m][j+n] * x[b][0][i+k][j+l]);
              }
            }
            atomicAdd(&grad_b[p], grad_output[b][0][i][j] * x[b][0][i+k][j+l]);
            int ii = i + 1 - k, jj = j + 1 - l;
            if (ii < 0 || ii >= N1 || jj < 0 || jj >= N2) continue;
            for (int m = 0; m <= 2; ++m) {
              for (int n = 0; n <= 2; ++n) {
                grad_x[b][0][i][j] += grad_output[b][0][ii][jj] * (weights[p][0][m][n] * image[0][ii+m][jj+n]);
              }
            }
            grad_x[b][0][i][j] += grad_output[b][0][ii][jj] * bias[p];
          }
        }
    //   }
    // }
  // }
}
} // namespace

std::vector<torch::Tensor> sm_block_cuda_forward(
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {

  const int B = x.size(0);
  const int N = x.size(2)-2;
  auto y = torch::zeros({B, x.size(1), N, N}, torch::dtype(x.dtype()).device(x.device()));

  const int nthreads = 1024; // for 2D num_threads = 32*32 = 1024
  const int nblocks = (B*N*N + nthreads - 1) / nthreads;

  const dim3 threads(nthreads);
  const dim3 blocks(nblocks);

  AT_DISPATCH_FLOATING_TYPES(x.type(), "sm_block_forward_cuda", ([&] {
    sm_block_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        y.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));

  return {y};
}

std::vector<torch::Tensor> sm_block_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {

  const int B = x.size(0);
  const int N = x.size(2)-2;

  const int nthreads = 256; // for 2D num_threads = 32*32 = 1024, not sure why 32 gives out of resources error
  const int nblocks = (B*N*N + nthreads - 1) / nthreads;

  const dim3 threads(nthreads);
  const dim3 blocks(nblocks);

  auto grad_x = torch::zeros({x.size(0), x.size(1), N, N}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_w = torch::zeros({9, 1, 3, 3}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_b = torch::zeros({9}, torch::dtype(x.dtype()).device(x.device()));

  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "sm_block_backward_cuda", ([&] {
    sm_block_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        grad_x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_w.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_b.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
  }));

  return {grad_x, grad_w, grad_b};
}