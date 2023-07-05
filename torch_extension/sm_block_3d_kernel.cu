#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define KERNEL_SIZE 27
__constant__ unsigned char WEIGHT_BYTES[KERNEL_SIZE * KERNEL_SIZE * sizeof(double)];
__constant__ unsigned char BIAS_BYTES[KERNEL_SIZE * sizeof(double)];

#define NUM_THREADS_FORWARD 512
#define NUM_THREADS_BACKWARD 256

namespace {
template <typename scalar_t>
__global__ void sm_block_3d_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image, // 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> x, // bs, 1, N, N, N
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> y) { // bs, 1, N, N, N

  const int B = x.size(0);
  const int N1 = image.size(1) - 2;
  const int N2 = image.size(2) - 2;
  const int N3 = image.size(3) - 2;

  const scalar_t *WEIGHT = (const scalar_t *)(WEIGHT_BYTES);
  const scalar_t *BIAS = (const scalar_t *)(BIAS_BYTES);

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const int ij = threadId % N3;
  threadId /= N3;
  const int j = threadId % N2;
  threadId /= N2;
  const int i = threadId % N1;
  const int b = threadId / N1;

  if (b < 0 || b >= B) return;

  scalar_t z = 0.0;
  for (int k = 0; k <= 2; ++k) {
    for (int l = 0; l <= 2; ++l) {
      for (int kl = 0; kl <= 2; ++kl) {
        int p = 3 * (3 * k + l) + kl;
        z = BIAS[p];
        for (int m = 0; m <= 2; ++m) {
          for (int n = 0; n <= 2; ++n) {
            for (int mn = 0; mn <= 2; ++mn) {
              z += (WEIGHT[27*p+9*m+3*n+mn] * image[0][i+m][j+n][ij+mn]);// * x[b][0][i+k][j+l][ij+kl];
            }
          }
        }
        z *= x[b][0][i+k][j+l][ij+kl];
        y[b][0][i][j][ij] += z;
      }
    }
  }
}


template <typename scalar_t>
__global__ void sm_block_3d_cuda_dwdb_fast_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_output, // bs, 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image, // 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> x, // bs, 1, N, N, N
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_w,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> grad_b,
    const int locationsPerBlock) {

  __shared__ scalar_t d_w[KERNEL_SIZE*KERNEL_SIZE];
  __shared__ scalar_t d_b[KERNEL_SIZE];

  const int B = x.size(0);
  const int N1 = (image.size(1) - 2) / locationsPerBlock;
  const int N2 = (image.size(2) - 2) / locationsPerBlock;
  const int N3 = (image.size(3) - 2) / locationsPerBlock;

  const int nBlocksPerCopy = ((KERNEL_SIZE+1)*KERNEL_SIZE + blockDim.x - 1) / blockDim.x;

  const int block = blockIdx.x / nBlocksPerCopy;
  const int innerBlock = blockIdx.x % nBlocksPerCopy;

  int location = block;
  const int ij = (location % N3) * locationsPerBlock;
  location /= N3;
  const int j = (location % N2) * locationsPerBlock;
  location /= N2;
  const int i = (location % N1) * locationsPerBlock;
  const int b = location / N1;


  const int p = innerBlock * blockDim.x + threadIdx.x;

  if (p >= (KERNEL_SIZE+1)*KERNEL_SIZE) return; // Wasted threads

  const int pp = p % KERNEL_SIZE;
  const int kl = pp % 3;
  const int l = (pp / 3) % 3;
  const int k = pp / 9;

  const int p0 = p / KERNEL_SIZE;

  int mn, n, m;
  int c;

  if (p0 == KERNEL_SIZE) {
    d_b[pp] = 0.0;
    for (int _i = 0; _i < locationsPerBlock; ++_i) {
      for (int _j = 0; _j < locationsPerBlock; ++_j) {
        for (int _ij = 0; _ij < locationsPerBlock; ++_ij) {
          d_b[pp] += grad_output[b][0][i+_i][j+_j][ij+_ij] * x[b][0][i+_i+k][j+_j+l][ij+_ij+kl];
        }
      }
    }
  } else {
    mn = p0 % 3;
    n = (p0 / 3) % 3;
    m = p0 / 9;
    c = 27*pp + 9*m + 3*n + mn;
    d_w[c] = 0.0;
    for (int _i = 0; _i < locationsPerBlock; ++_i) {
      for (int _j = 0; _j < locationsPerBlock; ++_j) {
        for (int _ij = 0; _ij < locationsPerBlock; ++_ij) {
          d_w[c] += grad_output[b][0][i+_i][j+_j][ij+_ij] * image[0][i+_i+m][j+_j+n][ij+_ij+mn] * x[b][0][i+_i+k][j+_j+l][ij+_ij+kl];
        }
      }
    }
  }

  if (p0 == KERNEL_SIZE)
    atomicAdd(&grad_b[pp], d_b[pp]);
  else
    atomicAdd(&grad_w[pp][0][m][n][mn], d_w[c]);
}


template <typename scalar_t>
__global__ void sm_block_3d_cuda_dx_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_output, // bs, 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image, // 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> x, // bs, 1, N, N, N
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_x) {

  const scalar_t *WEIGHT = (const scalar_t *)(WEIGHT_BYTES);
  const scalar_t *BIAS = (const scalar_t *)(BIAS_BYTES);

  const int B = x.size(0);
  const int N1 = image.size(1) - 2;
  const int N2 = image.size(2) - 2;
  const int N3 = image.size(3) - 2;

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId < B * N1 * N2 * N3) {
    const int ij = threadId % N3;

    threadId /= N3;
    const int j = threadId % N2;

    threadId /= N2;
    const int i = threadId % N1;

    threadId /= N1;
    const int b = threadId % B;

    scalar_t z = 0.0;
    for (int k = 0; k <= 2; ++k) {
      for (int l = 0; l <= 2; ++l) {
        for (int kl = 0; kl <= 2; ++kl) {
          int q = 3 * (3 * k + l) + kl;

          int ii = i + 1 - k, jj = j + 1 - l, iijj = ij + 1 - kl;
          if (ii < 0 || ii >= N1 || jj < 0 || jj >= N2 || iijj < 0 || iijj >= N3) continue;

          z = BIAS[q];
          for (int m = 0; m <= 2; ++m) {
            for (int n = 0; n <= 2; ++n) {
              for (int mn = 0; mn <= 2; ++mn) {
                z += WEIGHT[q*27+m*9+n*3+mn] * image[0][ii+m][jj+n][iijj+mn];
              }
            }
          }
          z *= grad_output[b][0][ii][jj][iijj];
          grad_x[b][0][i][j][ij] += z;
        }
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

  if (x.dtype() == torch::ScalarType::Double) {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<double>(), KERNEL_SIZE*KERNEL_SIZE * sizeof(double));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<double>(), KERNEL_SIZE * sizeof(double));
  } else {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<float>(), KERNEL_SIZE*KERNEL_SIZE * sizeof(float));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<float>(), KERNEL_SIZE * sizeof(float));
  }

  const int B = x.size(0);
  const int N = x.size(2)-2;
  auto y = torch::zeros({B, x.size(1), N, N, N}, torch::dtype(x.dtype()).device(x.device()));

  const int nthreads = NUM_THREADS_FORWARD; // for 2D num_threads = 32*32 = 1024
  const int nblocks = (B*N*N*N + nthreads - 1) / nthreads;

  const dim3 threads(nthreads);
  const dim3 blocks(nblocks);

  AT_DISPATCH_FLOATING_TYPES(x.type(), "sm_block_3d_forward_cuda", ([&] {
    sm_block_3d_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
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

  if (x.dtype() == torch::ScalarType::Double) {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<double>(), KERNEL_SIZE*KERNEL_SIZE * sizeof(double));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<double>(), KERNEL_SIZE * sizeof(double));
  } else {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<float>(), KERNEL_SIZE*KERNEL_SIZE * sizeof(float));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<float>(), KERNEL_SIZE * sizeof(float));
  }

  const int B = x.size(0);
  const int N = x.size(2)-2;
  const int totalData = B * N * N * N;

  const int nThreads = NUM_THREADS_BACKWARD;
  const int nBlocksPerCopy = ((KERNEL_SIZE+1)*KERNEL_SIZE + nThreads - 1) / nThreads;
  const int locationsPerBlock = 4;

  assert(N % locationsPerBlock == 0); // Data must be divisible by divisions

  const dim3 threads(nThreads);
  const dim3 blocks(nBlocksPerCopy*(totalData/ std::pow(locationsPerBlock, 3))), bblocks((totalData + nThreads - 1) / nThreads);

  auto grad_x = torch::zeros({B, x.size(1), N, N, N}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_w = torch::zeros({27, 1, 3, 3, 3}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_b = torch::zeros({27}, torch::dtype(x.dtype()).device(x.device()));

  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "sm_block_3d_cuda_dwdb", ([&] {
    sm_block_3d_cuda_dwdb_fast_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_w.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_b.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        locationsPerBlock);
  }));
  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "sm_block_3d_cuda_dx", ([&] {
    sm_block_3d_cuda_dx_kernel<scalar_t><<<bblocks, threads>>>(
        grad_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
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
