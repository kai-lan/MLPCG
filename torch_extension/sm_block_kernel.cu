#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmaxf(0.0, z) + fminf(0.0, alpha * (exp(z) - 1.0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}

template <typename scalar_t>
__global__ void sm_block_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> image, // 1, N, N
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> x, // bs, 1, N, N
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> weights, // 9, 1, 3, 3
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> bias, // 9
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> y) { // bs, 1, N, N
  //batch index
  // const int n = blockIdx.y;
  // column index
  // const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int N = image.size(2) - 2;
  const int bs = x.size(0);

  for (int b = 0; b < bs; ++b) {
    for (int i = 1; i <= N; ++i) {
      for (int j = 1; j <= N; ++j) {
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
      }
    }
  }
}

template <typename scalar_t>
__global__ void sm_block_cuda_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grad_output,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> image,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> x,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grad_w,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> grad_b) {
  //batch index
  // const int n = blockIdx.y;
  // column index
  // const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int N = image.size(2) - 2;
  const int bs = x.size(0);

  for (int b = 0; b < bs; ++b) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        for (int k = 0; k <= 2; ++k) {
          for (int l = 0; l <= 2; ++l) {
            int p = 3 * k + l;
            for (int m = 0; m <= 2; ++m) {
              for (int n = 0; n <= 2; ++n) {
                grad_w[p][0][m][n] += grad_output[b][0][i][j] * image[0][i+m][j+n] * x[b][0][i+k][j+l];
              }
            }
            grad_b[p] += grad_output[b][0][i][j] * x[b][0][i+k][j+l];
          }
        }
      }
    }

  }
}
} // namespace

std::vector<torch::Tensor> sm_block_cuda_forward(
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {

  auto y = torch::zeros({x.size(0), x.size(1), x.size(2)-2, x.size(3)-2}, torch::dtype(x.dtype()).device(x.device()));

  // const auto batch_size = old_cell.size(0);
  // const auto state_size = old_cell.size(1);

  const int threads = 1;
  const int blocks = 1;
  // const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(x.type(), "sm_block_forward_cuda", ([&] {
    sm_block_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        x.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        weights.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        bias.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        y.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>());
  }));

  return {y};
}

std::vector<torch::Tensor> sm_block_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor image,
    torch::Tensor x) {

  const int threads = 1;
  const int blocks = 1;
  // const dim3 blocks((state_size + threads - 1) / threads, batch_size);
  auto grad_w = torch::zeros({9, 1, 3, 3}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_b = torch::zeros({9}, torch::dtype(x.dtype()).device(x.device()));

  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "sm_block_backward_cuda", ([&] {
    sm_block_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        image.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        x.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        grad_w.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        grad_b.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>());
  }));

  // auto d_gate_weights = d_gates.flatten(1, 2);
  // auto d_weights = d_gate_weights.t().mm(X);
  // auto d_bias = d_gate_weights.sum(/*dim=*/0, /*keepdim=*/true);

  // auto d_X = d_gate_weights.mm(weights);
  // auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  // auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {grad_w, grad_b};
}