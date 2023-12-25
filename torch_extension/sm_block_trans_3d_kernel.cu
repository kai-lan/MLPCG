#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define NUM_IMAGES 3
#define KERNEL_SIZE 27
#define WEIGHT_SIZE NUM_IMAGES*KERNEL_SIZE*KERNEL_SIZE
__constant__ unsigned char WEIGHT_BYTES[WEIGHT_SIZE*sizeof(double)];
__constant__ unsigned char BIAS_BYTES[KERNEL_SIZE*sizeof(double)];

#define LOCATIONS_PER_BLOCK 8
#define NUM_THREADS_FORWARD 256

namespace {
// template <typename scalar_t>
// __global__ void sm_block_trans_3d_cuda_forward_kernel(
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image, // num_imgs, N, N
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> x, // bs, 1, N, N
//     torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> y,
//     const int B, const int N1, const int N2, const int N3) { // bs, 1, N, N, N

//   const scalar_t *WEIGHT = (const scalar_t *)(WEIGHT_BYTES);
//   const scalar_t *BIAS = (const scalar_t *)(BIAS_BYTES);

//   int threadId = blockIdx.x * blockDim.x + threadIdx.x;
//   if (threadId >= B*N1*N2*N3) return;
//   const int ij = threadId % N3;
//   threadId /= N3;
//   const int j = threadId % N2;
//   threadId /= N2;
//   const int i = threadId % N1;
//   const int c = threadId / N1;

//   scalar_t zz = 0.0;
//   for (int k = 0; k <= 2; ++k) {
//     int alpha = i+k;
//     for (int l = 0; l <= 2; ++l) {
//       int beta = j+l;
//       for (int kl = 0; kl <= 2; ++kl) {
//         // int ii = i + k - 1, jj = j + l - 1, iijj = ij + kl - 1;
//         // if (ii < 0 || ii >= N1) continue;
//         // if (jj < 0 || jj >= N2) continue;
//         // if (iijj < 0 || iijj >= N3) continue;
//         int gamma = ij+kl;
//         int p = 3 * (3 * k + l) + kl;
//         scalar_t K_ij = BIAS[p];
//         for (int s = 0; s < NUM_IMAGES; ++s) {
//           for (int m = 0; m <= 2; ++m) {
//             int aa = i+m;
//             for (int n = 0; n <= 2; ++n) {
//               int bb = j+n;
//               for (int mn = 0; mn <=2; ++mn) {
//                 int cc = ij+mn;
//                 K_ij += WEIGHT[3*(3*(3*(NUM_IMAGES*p+s)+m)+n)+mn] * image[s][aa][bb][cc];
//               }
//             }
//           }
//         }
//         // scalar_t val = K_ij * x[c][0][i+k][j+l][ij+kl];
//         zz += K_ij * x[c][0][alpha][beta][gamma];
//         // atomicAdd(&y[c][0][i][j], val);
//         // scalar_t val = K_ij * x[c][0][i+1][j+1][ij+1];
//         // atomicAdd(&y[c][0][ii][jj][iijj], val);
//         // y[c][0][ii][jj] += K_ij * x[c][0][i+1][j+1];
//       }
//     }
//   }
//   y[c][0][i][j][ij] = zz;
// } // forward kernel

template <typename scalar_t>
__global__ void sm_block_trans_3d_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image, // num_imgs, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> x, // bs, 1, N, N, N
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> y,
    const int B, const int N1, const int N2, const int N3) { // bs, 1, N, N, N

  const scalar_t *WEIGHT = (const scalar_t *)(WEIGHT_BYTES);
  const scalar_t *BIAS = (const scalar_t *)(BIAS_BYTES);

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadId >= B*N1*N2*N3) return;
  const int ij = threadId % N3;
  threadId /= N3;
  const int j = threadId % N2;
  threadId /= N2;
  const int i = threadId % N1;
  const int c = threadId / N1;

  scalar_t zz = 0.0;
  for (int k = 0; k <= 2; ++k) {
    int alpha = i+k;
    for (int l = 0; l <= 2; ++l) {
      int beta = j+l;
      for (int kl = 0; kl <= 2; ++kl) {
        int gamma = ij+kl;
        int p = 3 * (3 * k + l) + kl;
        scalar_t z = BIAS[p];
        for (int s = 0; s < NUM_IMAGES; ++s) {
          for (int m = 0; m <= 2; ++m) {
            int aa = i+m;
            for (int n = 0; n <= 2; ++n) {
              int bb = j+n;
              for (int mn = 0; mn <= 2; ++mn) {
                int cc = ij+mn;
                z += (WEIGHT[3*(3*(3*(NUM_IMAGES*p+s)+m)+n)+mn] * image[s][aa][bb][cc]);
              }
            }
          }
        }
        z *= x[c][0][alpha][beta][gamma];
        zz += z;
      }
    }
  }
  y[c][0][i][j][ij] = zz;
} // forward kernel

} // namespace

std::vector<torch::Tensor> sm_block_trans_3d_cuda_forward(
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {

  assert(image.size(0) == NUM_IMAGES);

  if (x.dtype() == torch::ScalarType::Double) {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<double>(), WEIGHT_SIZE * sizeof(double));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<double>(), KERNEL_SIZE * sizeof(double));
  } else {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<float>(), WEIGHT_SIZE * sizeof(float));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<float>(), KERNEL_SIZE * sizeof(float));
  }

  const int B = x.size(0);
  const int N1 = image.size(1)-2;
  const int N2 = image.size(2)-2;
  const int N3 = image.size(3)-2;
  auto y = torch::zeros({B, x.size(1), N1, N2, N3}, torch::dtype(x.dtype()).device(x.device()));

  const int nThreads = NUM_THREADS_FORWARD;
  const int nBlocks = (B*N1*N2*N3 + nThreads - 1) / nThreads;

  const dim3 threads(nThreads);
  const dim3 blocks(nBlocks);

  AT_DISPATCH_FLOATING_TYPES(x.type(), "sm_block_trans_3d_forward_cuda", ([&] {
    sm_block_trans_3d_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        y.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        B, N1, N2, N3);
  }));

  return {y};
}
