#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define KERNEL_SIZE 27
__constant__ unsigned char WEIGHT_BYTES[KERNEL_SIZE * KERNEL_SIZE * sizeof(double)];
__constant__ unsigned char BIAS_BYTES[KERNEL_SIZE * sizeof(double)];

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
  if (i < 0 || i >= N1) return;
  if (j < 0 || j >= N2) return;
  if (ij < 0 || ij >= N3) return;

  scalar_t z = 0.0;
  for (int k = 0; k <= 2; ++k) {
    for (int l = 0; l <= 2; ++l) {
      for (int kl = 0; kl <= 2; ++kl) {
        int p = 3 * (3 * k + l) + kl;
        z = BIAS[p];
        // z = bias[p];
        for (int m = 0; m <= 2; ++m) {
          for (int n = 0; n <= 2; ++n) {
            for (int mn = 0; mn <= 2; ++mn) {
              z += (WEIGHT[27*p+9*m+3*n+mn] * image[0][i+m][j+n][ij+mn]);// * x[b][0][i+k][j+l][ij+kl];
              // z += (weights[p][0][m][n][mn] * image[0][i+m][j+n][ij+mn]);// * x[b][0][i+k][j+l][ij+kl];
            }
          }
        }
        z *= x[b][0][i+k][j+l][ij+kl];
        y[b][0][i][j][ij] += z;
        // y[b][0][i][j][ij] += bias[p] * x[b][0][i+k][j+l][ij+kl];
      }
    }
  }
}

// template <typename scalar_t>
// __global__ void sm_block_3d_cuda_backward_kernel(
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_output, // bs, 1, N, N, N
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image, // 1, N, N, N
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> x, // bs, 1, N, N, N
//     torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_x,
//     torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_w,
//     torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> grad_b) {

//   __shared__ scalar_t d_w[KERNEL_SIZE*KERNEL_SIZE];
//   __shared__ scalar_t d_b[KERNEL_SIZE];

//   const scalar_t *WEIGHT = (const scalar_t *)(WEIGHT_BYTES);
//   const scalar_t *BIAS = (const scalar_t *)(BIAS_BYTES);

//   const int B = x.size(0);
//   const int N1 = image.size(1) - 2;
//   const int N2 = image.size(2) - 2;
//   const int N3 = image.size(3) - 2;

//   int threadId = blockIdx.x * blockDim.x + threadIdx.x;

//   int id = threadIdx.x;
//   while (id < KERNEL_SIZE*KERNEL_SIZE) {
//     d_w[id] = 0.0;
//     id += blockDim.x;
//   }

//   if (threadIdx.x >= blockDim.x - KERNEL_SIZE && threadIdx.x < blockDim.x) {
//     d_b[threadIdx.x - (blockDim.x - KERNEL_SIZE)] = 0.0;
//   }
//   __syncthreads();

//   if (threadId < B * N1 * N2 * N3) {
//     const int ij = threadId % N3;

//     threadId /= N3;
//     const int j = threadId % N2;

//     threadId /= N2;
//     const int i = threadId % N1;

//     threadId /= N1;
//     const int b = threadId % B;

//     for (int k = 0; k <= 2; ++k) {
//       for (int l = 0; l <= 2; ++l) {
//         for (int kl = 0; kl <= 2; ++kl) {
//           int q = 3 * (3 * k + l) + kl;
//           for (int m = 0; m <= 2; ++m) {
//             for (int n = 0; n <= 2; ++n) {
//               for (int mn = 0; mn <= 2; ++mn) {
//                 // atomicAdd(&grad_w[q][0][m][n][mn], grad_output[b][0][i][j][ij] * image[0][i+m][j+n][ij+mn] * x[b][0][i+k][j+l][ij+kl]);
//                 atomicAdd(&d_w[27*q+9*m+3*n+mn], grad_output[b][0][i][j][ij] * image[0][i+m][j+n][ij+mn] * x[b][0][i+k][j+l][ij+kl]);
//               }
//             }
//           }
//           // atomicAdd(&grad_b[q], grad_output[b][0][i][j][ij] * x[b][0][i+k][j+l][ij+kl]);
//           atomicAdd(&d_b[q], grad_output[b][0][i][j][ij] * x[b][0][i+k][j+l][ij+kl]);

//           int ii = i + 1 - k, jj = j + 1 - l, iijj = ij + 1 - kl;
//           if (ii < 0 || ii >= N1 || jj < 0 || jj >= N2 || iijj < 0 || iijj >= N3) continue;
//           for (int m = 0; m <= 2; ++m) {
//             for (int n = 0; n <= 2; ++n) {
//               for (int mn = 0; mn <= 2; ++mn) {
//                 grad_x[b][0][i][j][ij] += grad_output[b][0][ii][jj][iijj] * (WEIGHT[q*27+m*9+n*3+mn] * image[0][ii+m][jj+n][iijj+mn]);
//               }
//             }
//           }
//           grad_x[b][0][i][j][ij] += grad_output[b][0][ii][jj][iijj] * BIAS[q];
//         }
//       }
//     }
//   }
//   __syncthreads();

//   // Only assign some threads to do this
//   if (threadIdx.x >= blockDim.x - KERNEL_SIZE && threadIdx.x < blockDim.x) {
//     int q = threadIdx.x - (blockDim.x - KERNEL_SIZE);
//     atomicAdd(&grad_b[q], d_b[q]);
//   }

//   id = threadIdx.x;
//   while (id < KERNEL_SIZE*KERNEL_SIZE) {
//     int p = id;
//     int mn = p % 3;
//     p /= 3;
//     int n = p % 3;
//     p /= 3;
//     int m = p % 3;
//     int q = p / 3;

//     atomicAdd(&grad_w[q][0][m][n][mn], d_w[27*q+9*m+3*n+mn]);

//     id += blockDim.x;
//   }

// }


template <typename scalar_t>
__global__ void sm_block_3d_cuda_dwdb_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_output, // bs, 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image, // 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> x, // bs, 1, N, N, N
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_x,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_w,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> grad_b,
    const int nBlocksPerElement) {

  __shared__ scalar_t val[512];

  const int B = x.size(0);
  const int N1 = image.size(1) - 2;
  const int N2 = image.size(2) - 2;
  const int N3 = image.size(3) - 2;

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  int location = threadId % (nBlocksPerElement * blockDim.x);

  bool withinRange = location < (B*N1*N2*N3);
  // (27+1)*27, B, N1, N2, N3
  // threadId = (((p * B + b) * N1 + i) * N2 + j) * N3 + ij
  const int ij = location % N3;

  location /= N3;
  const int j = location % N2;

  location /= N2;
  const int i = location % N1;

  location /= N1;
  const int b = location % B;

  const int p = threadId / (nBlocksPerElement * blockDim.x);

  int mn, n, m, pp, kl, l, k;

  if (!withinRange) {
    val[threadIdx.x] = 0.0;
  } else {
    pp = p % KERNEL_SIZE;
    kl = pp % 3;
    l = (pp / 3) % 3;
    k = pp / 9;
    // p /= KERNEL_SIZE;
    const int p0 = p / KERNEL_SIZE;
    if (p0 == KERNEL_SIZE) {
      // atomicAdd(&grad_b[pp], grad_output[b][0][i][j][ij] * x[b][0][i+k][j+l][ij+kl]);
      // atomicAdd(&val, grad_output[b][0][i][j][ij] * x[b][0][i+k][j+l][ij+kl]);
      val[threadIdx.x] = grad_output[b][0][i][j][ij] * x[b][0][i+k][j+l][ij+kl];

    } else {
      mn = p0 % 3;
      n = (p0 / 3) % 3;
      m = p0 / 9;

      // atomicAdd(&grad_w[pp][0][m][n][mn], grad_output[b][0][i][j][ij] * image[0][i+m][j+n][ij+mn] * x[b][0][i+k][j+l][ij+kl]);
      // atomicAdd(&val, grad_output[b][0][i][j][ij] * image[0][i+m][j+n][ij+mn] * x[b][0][i+k][j+l][ij+kl]);
      val[threadIdx.x] = grad_output[b][0][i][j][ij] * image[0][i+m][j+n][ij+mn] * x[b][0][i+k][j+l][ij+kl];
    }
  }
  __syncthreads();

  // Reduction
  int data = blockDim.x; // 512 -> 256 -> 128
  while(data > 1) {
    if (threadIdx.x < data / 2)
      val[threadIdx.x] += val[threadIdx.x + data / 2];
    data /= 2;
    __syncthreads();
  }

  if (threadIdx.x > 0) return;
  if (p / KERNEL_SIZE == KERNEL_SIZE)
    atomicAdd(&grad_b[pp], val[0]);
  else
    atomicAdd(&grad_w[pp][0][m][n][mn], val[0]);
}

template <typename scalar_t>
__global__ void sm_block_3d_cuda_dx_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_output, // bs, 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image, // 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> x, // bs, 1, N, N, N
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_x,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_w,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> grad_b) {

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

  const int nthreads = 512; // for 2D num_threads = 32*32 = 1024
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

  const int nThreads = 512; // for 2D num_threads = 32*32 = 1024, not sure why 32 gives out of resources error
  const int nBlocksPerElement = (totalData + nThreads - 1) / nThreads; // number of blocks one element dW_{p, 0, m, n, mn} or db_{p}
  const int nBlocks = nBlocksPerElement * (KERNEL_SIZE + 1) * KERNEL_SIZE; // dw size 27 * 27 and db size 27
  // const int nBlocks = (totalData + nThreads - 1) / nThreads;

  const dim3 threads(nThreads);
  const dim3 blocks(nBlocks), bblocks(nBlocksPerElement);

  auto grad_x = torch::zeros({B, x.size(1), N, N, N}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_w = torch::zeros({27, 1, 3, 3, 3}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_b = torch::zeros({27}, torch::dtype(x.dtype()).device(x.device()));

  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "sm_block_3d_backward_cuda", ([&] {
    sm_block_3d_cuda_dwdb_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_w.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_b.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        nBlocksPerElement);
  }));
  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "sm_block_3d_backward_cuda", ([&] {
    sm_block_3d_cuda_dx_kernel<scalar_t><<<bblocks, threads>>>(
        grad_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
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
