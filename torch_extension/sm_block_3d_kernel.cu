#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
template <typename scalar_t>
__global__ void sm_block_3d_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image, // 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> x, // bs, 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> weights, // 27, 1, 3, 3, 3
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> bias, // 27
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> y) { // bs, 1, N, N, N

  const int B = x.size(0);
  const int N1 = image.size(1) - 2;
  const int N2 = image.size(2) - 2;
  const int N3 = image.size(3) - 2;

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const int ij = threadId % N3;
  threadId /= N3;
  const int j = threadId % N2;
  threadId /= N2;
  const int i = threadId % N1;
  const int b = threadId / N1;

  if (b < 0 || b >= B) return;
  if (i < 0 || i >= N1) return;
  if (j < 0 || j >= N2) return;
  if (ij < 0 || ij >= N3) return;

  scalar_t z = 0.0;
  for (int k = 0; k <= 2; ++k) {
    for (int l = 0; l <= 2; ++l) {
      for (int kl = 0; kl <= 2; ++kl) {
        int p = 3 * (3 * k + l) + kl;
        z = bias[p];
        for (int m = 0; m <= 2; ++m) {
          for (int n = 0; n <= 2; ++n) {
            for (int mn = 0; mn <= 2; ++mn) {
              z += (weights[p][0][m][n][mn] * image[0][i+m][j+n][ij+mn]);
            }
          }
        }
        z *= x[b][0][i+k][j+l][ij+kl];
        // z += bias[p] * x[b][0][i+k][j+l][ij+kl];
        y[b][0][i][j][ij] += z;
      }
    }
  }
}

template <typename scalar_t>
__global__ void sm_block_3d_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_output, // bs, 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image, // 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> x, // bs, 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> weights, // 27, 1, 3, 3, 3
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> bias, // 27
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_x,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_w,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> grad_b) {

  const int B = x.size(0);
  const int N1 = image.size(1) - 2;
  const int N2 = image.size(2) - 2;
  const int N3 = image.size(3) - 2;

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const int ij = threadId % N3;
  threadId /= N3;
  const int j = threadId % N2;
  threadId /= N2;
  const int i = threadId % N1;
  const int b = threadId / N1;

  if (b < 0 || b >= B) return;
  if (i < 0 || i >= N1) return;
  if (j < 0 || j >= N2) return;
  if (ij < 0 || ij >= N3) return;

  for (int k = 0; k <= 2; ++k) {
    for (int l = 0; l <= 2; ++l) {
      for (int kl = 0; kl <= 2; ++kl) {
        int p = 3 * (3 * k + l) + kl;
        for (int m = 0; m <= 2; ++m) {
          for (int n = 0; n <= 2; ++n) {
            for (int mn = 0; mn <= 2; ++mn) {
              atomicAdd(&grad_w[p][0][m][n][mn], grad_output[b][0][i][j][ij] * image[0][i+m][j+n][ij+mn] * x[b][0][i+k][j+l][ij+kl]);
            }
          }
        }
        atomicAdd(&grad_b[p], grad_output[b][0][i][j][ij] * x[b][0][i+k][j+l][ij+kl]);

        int ii = i + 1 - k, jj = j + 1 - l, iijj = ij + 1 - kl;
        if (ii < 0 || ii >= N1 || jj < 0 || jj >= N2 || iijj < 0 || iijj >= N3) continue;
        for (int m = 0; m <= 2; ++m) {
          for (int n = 0; n <= 2; ++n) {
            for (int mn = 0; mn <= 2; ++mn) {
              grad_x[b][0][i][j][ij] += grad_output[b][0][ii][jj][iijj] * (weights[p][0][m][n][mn] * image[0][ii+m][jj+n][iijj+mn]);
            }
          }
        }
        grad_x[b][0][i][j][ij] += grad_output[b][0][ii][jj][iijj] * bias[p];
      }
    }
  }
}
} // namespace

std::vector<torch::Tensor> sm_block_3d_cuda_forward(
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {

  const int B = x.size(0);
  const int N = x.size(2)-2;
  auto y = torch::zeros({B, x.size(1), N, N, N}, torch::dtype(x.dtype()).device(x.device()));

  const int nthreads = 1024; // for 2D num_threads = 32*32 = 1024
  const int nblocks = (B*N*N*N + nthreads - 1) / nthreads;

  const dim3 threads(nthreads);
  const dim3 blocks(nblocks);

  AT_DISPATCH_FLOATING_TYPES(x.type(), "sm_block_3d_forward_cuda", ([&] {
    sm_block_3d_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        y.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
  }));

  return {y};
}

std::vector<torch::Tensor> sm_block_3d_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {

  const int B = x.size(0);
  const int N = x.size(2)-2;

  const int nthreads = 512; // for 2D num_threads = 32*32 = 1024, not sure why 32 gives out of resources error
  const int nblocks = (B*N*N*N + nthreads - 1) / nthreads;

  const dim3 threads(nthreads);
  const dim3 blocks(nblocks);

  auto grad_x = torch::zeros({B, x.size(1), N, N, N}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_w = torch::zeros({27, 1, 3, 3, 3}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_b = torch::zeros({27}, torch::dtype(x.dtype()).device(x.device()));

  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "sm_block_3d_backward_cuda", ([&] {
    sm_block_3d_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        grad_x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_w.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_b.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
  }));

  return {grad_x, grad_w, grad_b};
}

int main() {

  int N = 12;
  auto image = torch::ones({1, N, N, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto x = torch::rand({5, 1, N, N, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto weight = torch::rand({27, 1, 3, 3, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto bias = torch::rand({27}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  // curandState *d_states;
  // cudaMalloc(&d_states, CNT * sizeof(curandState));
  // kernel_setup_randstates_2d<<<1,CNT>>>(d_states, 1,1, 1);
  auto y = sm_block_3d_cuda_forward(image, x, weight, bias);
  std::cout << y  << std::endl;
  cudaDeviceSynchronize();
}
