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

  const int N = image.size(2) - 2;
  const int bs = x.size(0);
  const int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (i < 1 || i > N) return;
  if (j < 1 || j > N) return;

  for (int b = 0; b < bs; ++b) {
    // for (int i = 1; i <= N; ++i) {
    //   for (int j = 1; j <= N; ++j) {
        for (int m = 0; m <= 2; ++m) {
          for (int n = 0; n <= 2; ++n) {
            int p = 3 * m + n;
            for (int k = -1; k <= 1; ++k) {
              for (int l = -1; l <= 1; ++l) {
                // K[m][n] += weights[p][0][k+1][l+1] * image[0][i+k][j+l] + bias[p];
                y[b][0][i-1][j-1] += (weights[p][0][k+1][l+1] * image[0][i+k][j+l]) * x[b][0][i+m-1][j+n-1];
              }
            }
            y[b][0][i-1][j-1] += bias[p] * x[b][0][i+m-1][j+n-1];
            // y[b][0][i][j] += K[m][n] * x[b][0][i+m-1][j+n-1];
          }
        }
    //   }
    // }
  }
}

template <typename scalar_t>
__global__ void sm_block_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> image,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_w,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> grad_b) {

  const int N = image.size(2) - 2;
  const int bs = x.size(0);
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < 0 || i >= N) return;
  if (j < 0 || j >= N) return;

  for (int b = 0; b < bs; ++b) {
    // for (int i = 0; i < N; ++i) {
    //   for (int j = 0; j < N; ++j) {
        for (int k = 0; k <= 2; ++k) {
          for (int l = 0; l <= 2; ++l) {
            int p = 3 * k + l;
            for (int m = 0; m <= 2; ++m) {
              for (int n = 0; n <= 2; ++n) {
                atomicAdd(&grad_w[p][0][m][n], grad_output[b][0][i][j] * image[0][i+m][j+n] * x[b][0][i+k][j+l]);
                // grad_w[p][0][m][n] += grad_output[b][0][i][j] * image[0][i+m][j+n] * x[b][0][i+k][j+l];
              }
            }
            atomicAdd(&grad_b[p], grad_output[b][0][i][j] * x[b][0][i+k][j+l]);
            // grad_b[p] += grad_output[b][0][i][j] * x[b][0][i+k][j+l];
          }
        }
    //   }
    // }
  }
}
} // namespace

std::vector<torch::Tensor> sm_block_cuda_forward(
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {

  const int N = x.size(2)-2;
  auto y = torch::zeros({x.size(0), x.size(1), N, N}, torch::dtype(x.dtype()).device(x.device()));


  const int nthreads = 32; // for 2D num_threads = 32*32 = 1024
  const int nblocks = (N + nthreads - 1) / nthreads;
  const dim3 threads(nthreads, nthreads);
  const dim3 blocks(nblocks, nblocks);

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
    torch::Tensor x) {

  const int N = x.size(2)-2;
  const int nthreads = 32; // for 2D num_threads = 32*32 = 1024
  const int nblocks = (N + nthreads - 1) / nthreads;
  const dim3 threads(nthreads, nthreads);
  const dim3 blocks(nblocks, nblocks);

  auto grad_w = torch::zeros({9, 1, 3, 3}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_b = torch::zeros({9}, torch::dtype(x.dtype()).device(x.device()));

  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "sm_block_backward_cuda", ([&] {
    sm_block_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_w.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_b.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
  }));

  // auto d_gate_weights = d_gates.flatten(1, 2);
  // auto d_weights = d_gate_weights.t().mm(X);
  // auto d_bias = d_gate_weights.sum(/*dim=*/0, /*keepdim=*/true);

  // auto d_X = d_gate_weights.mm(weights);
  // auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  // auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {grad_w, grad_b};
}