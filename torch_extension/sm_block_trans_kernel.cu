#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define NUM_IMAGES 3
#define KERNEL_SIZE 9
#define WEIGHT_SIZE NUM_IMAGES*KERNEL_SIZE*KERNEL_SIZE
__constant__ unsigned char WEIGHT_BYTES[WEIGHT_SIZE*sizeof(double)];
__constant__ unsigned char BIAS_BYTES[KERNEL_SIZE*sizeof(double)];

#define LOCATIONS_PER_BLOCK 16
#define NUM_THREADS_FORWARD 256
#define NUM_THREADS_BACKWARD 256

namespace {
template <typename scalar_t>
__global__ void sm_block_trans_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> image, // num_imgs, N, N
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> x, // bs, 1, N, N
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> y,
    const int B, const int N1, const int N2) { // bs, 1, N, N

  const scalar_t *WEIGHT = (const scalar_t *)(WEIGHT_BYTES);
  const scalar_t *BIAS = (const scalar_t *)(BIAS_BYTES);

  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadId >= B*N1*N2) return;
  const int j = threadId % N2;
  const int i = (threadId/N2) % N1;
  const int c = (threadId/N2) / N1;

  scalar_t zz = 0.0;
  #pragma unroll
  for (int k = 0; k <= 2; ++k) {
    int ii = i + 1 - k;
    // int ii = i + k - 1;
    if (ii < 0 || ii >= N1) continue;
    for (int l = 0; l <= 2; ++l) {
      int jj = j + 1 - l;
      // int jj = j + l - 1;
      if (jj < 0 || jj >= N2) continue;
      int p = 3 * k + l;
      scalar_t K_ij = BIAS[p];
      #pragma unroll
      for (int s = 0; s < NUM_IMAGES; ++s) {
        for (int m = 0; m <= 2; ++m) {
          for (int n = 0; n <= 2; ++n) {
            K_ij += WEIGHT[3*(3*(NUM_IMAGES*p+s)+m)+n] * image[s][ii+m][jj+n];
            // K_ij += WEIGHT[3*(3*(NUM_IMAGES*p+s)+m)+n] * image[s][i+m][j+n];
          }
        }
      }
      zz += K_ij * x[c][0][ii+1][jj+1];
      // scalar_t val = K_ij * x[c][0][i+1][j+1];
      // atomicAdd(&y[c][0][ii][jj], val);
    }
  }
  y[c][0][i][j] = zz;

} // forward kernel

template <typename scalar_t>
__global__ void sm_block_trans_cuda_dwdb_fast_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_output, // bs, 1, N, N
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> image, // 1, N, N
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> x, // bs, 1, N, N
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_w,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> grad_b,
    const int nBlocksPerCopy, const int B, const int n1, const int n2) {

  __shared__ scalar_t OUT[LOCATIONS_PER_BLOCK+2][LOCATIONS_PER_BLOCK+2]; // 6^2 = 36
  __shared__ scalar_t I[NUM_IMAGES][LOCATIONS_PER_BLOCK+2][LOCATIONS_PER_BLOCK+2]; // 3 * 6^2 = 108
  __shared__ scalar_t X[LOCATIONS_PER_BLOCK][LOCATIONS_PER_BLOCK]; // 4^2 = 16

  const int N1 = LOCATIONS_PER_BLOCK * n1;
  const int N2 = LOCATIONS_PER_BLOCK * n2;

  const int location = blockIdx.x / nBlocksPerCopy;
  const int innerBlock = blockIdx.x % nBlocksPerCopy;

  const int j = (location % n2) * LOCATIONS_PER_BLOCK;
  const int i = ((location/n2) % n1) * LOCATIONS_PER_BLOCK;
  const int c = (location/n2) / n1;

  const int width = LOCATIONS_PER_BLOCK+2;
  const int I_Size = NUM_IMAGES*width*width; // 108
  const int OUT_Size = width*width; // 36
  const int X_Size = LOCATIONS_PER_BLOCK*LOCATIONS_PER_BLOCK; // 16

  int tid = threadIdx.x;
  while (tid < I_Size + X_Size + OUT_Size) {
    int ttid = tid;
    if (ttid < I_Size) {
      int iy = tid % width;
      tid /= width;
      int ix = tid % width;
      int s = tid / width;
      I[s][ix][iy] = image[s][i+ix][j+iy];
    }
    else if (ttid < I_Size + OUT_Size) {
      tid -= I_Size;
      int iy = tid % width;
      int ix = tid / width;
      // i + k - 1 = 0, 1, ..., N-1
      int ii = i + ix - 1, jj = j + iy - 1;
      if (ii < 0 || ii >= N1 || jj < 0 || jj >= N2)
        OUT[ix][iy] = 0.0;
      else
        OUT[ix][iy] = grad_output[c][0][ii][jj];
    }
    else {
      tid -= I_Size + OUT_Size;
      int iy = tid % LOCATIONS_PER_BLOCK;
      int ix = tid / LOCATIONS_PER_BLOCK;
      X[ix][iy] = x[c][0][i+ix+1][j+iy+1];
    }

    tid = ttid + blockDim.x;
  }

  const int p = innerBlock * blockDim.x + threadIdx.x;
  if (p >= WEIGHT_SIZE + KERNEL_SIZE) return; // Wasted threads

  const int idx_kl = p / (NUM_IMAGES*KERNEL_SIZE + 1);
  const int l = idx_kl % 3;
  const int k = idx_kl / 3;

  __syncthreads();

  const int idx_smn = p % (NUM_IMAGES*KERNEL_SIZE + 1);

  int s, m, n;
  int idx;

  scalar_t d_w = 0.0, d_b = 0.0;
  if (idx_smn == NUM_IMAGES * KERNEL_SIZE) {
    #pragma unroll
    for (int _i = 0; _i < LOCATIONS_PER_BLOCK; ++_i) {
      for (int _j = 0; _j < LOCATIONS_PER_BLOCK; ++_j) {
        d_b += OUT[_i+k][_j+l] * X[_i][_j];
      }
    }
  } else {
    n = idx_smn % 3;
    m = (idx_smn/3) % 3;
    s = (idx_smn/3) / 3;
    idx = KERNEL_SIZE*NUM_IMAGES*idx_kl + idx_smn;

    #pragma unroll
    for (int _i = 0; _i < LOCATIONS_PER_BLOCK; ++_i) {
      for (int _j = 0; _j < LOCATIONS_PER_BLOCK; ++_j) {
        d_w += OUT[_i+k][_j+l] * I[s][_i+m][_j+n] * X[_i][_j];
      }
    }
  }

  if (idx_smn == NUM_IMAGES * KERNEL_SIZE)
    atomicAdd(&grad_b[idx_kl], d_b);
  else
    atomicAdd(&grad_w[idx_kl][s][m][n], d_w);
} // backward dw, db

template <typename scalar_t>
__global__ void sm_block_trans_cuda_dx_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_output, // bs, 1, N, N
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> image, // num_imgs, N, N
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> x, // bs, 1, N, N
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_x,
    const int B, const int N1, const int N2) {

  const scalar_t *WEIGHT = (const scalar_t *)(WEIGHT_BYTES);
  const scalar_t *BIAS = (const scalar_t *)(BIAS_BYTES);

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId < B * N1 * N2) {
    const int j = threadId % N2;
    threadId /= N2;
    const int i = threadId % N1;
    threadId /= N1;
    const int c = threadId % B;

    scalar_t zz = 0.0;
    #pragma unroll
    for (int k = 0; k <= 2; ++k) {
      int ii = i + k - 1;
      if (ii < 0 || ii >= N1) continue;
      for (int l = 0; l <= 2; ++l) {
        int jj = j + l - 1;
        if (jj < 0 || jj >= N2) continue;

        int idx_kl = 3 * k + l;

        scalar_t z = BIAS[idx_kl];
        #pragma unroll
        for (int s = 0; s < NUM_IMAGES; ++s) {
          for (int m = 0; m <= 2; ++m) {
            for (int n = 0; n <= 2; ++n) {
              z += WEIGHT[3*(3*(NUM_IMAGES*idx_kl+s)+m)+n] * image[s][i+m][j+n];
            }
          }
        }
        z *= grad_output[c][0][ii][jj];
        zz += z;

      }
    }
    grad_x[c][0][i][j] = zz;
  }
} // backward dx
} // namespace

std::vector<torch::Tensor> sm_block_trans_cuda_forward(
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
  auto y = torch::zeros({B, x.size(1), N1, N2}, torch::dtype(x.dtype()).device(x.device()));

  const int nThreads = NUM_THREADS_FORWARD;
  const int nBlocks = (B*N1*N2 + nThreads - 1) / nThreads;

  const dim3 threads(nThreads);
  const dim3 blocks(nBlocks);

  AT_DISPATCH_FLOATING_TYPES(x.type(), "sm_block_trans_forward_cuda", ([&] {
    sm_block_trans_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        y.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        B, N1, N2);
  }));

  return {y};
}

std::vector<torch::Tensor> sm_block_trans_cuda_backward(
    torch::Tensor grad_output,
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
  const int totalData = B * N1 * N2;

  const int nThreads = NUM_THREADS_BACKWARD;
  const int nBlocksPerCopy = (WEIGHT_SIZE + KERNEL_SIZE + nThreads - 1) / nThreads;
  const int locationsPerBlock = LOCATIONS_PER_BLOCK;

  assert((N1 % locationsPerBlock == 0) && (N2 % locationsPerBlock == 0)); // Data must be divisible by divisions

  const int n1 = N1 / locationsPerBlock;
  const int n2 = N2 / locationsPerBlock;
  const dim3 threads(nThreads);
  const dim3 dwdb_blocks(nBlocksPerCopy*(B*n1*n2)), dx_blocks((totalData + nThreads - 1) / nThreads);

  auto grad_x = torch::zeros({B, x.size(1), N1, N2}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_w = torch::zeros({9, NUM_IMAGES, 3, 3}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_b = torch::zeros({9}, torch::dtype(x.dtype()).device(x.device()));

  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "sm_block_trans_cuda_dwdb", ([&] {
    sm_block_trans_cuda_dwdb_fast_kernel<scalar_t><<<dwdb_blocks, threads>>>(
        grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_w.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_b.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        nBlocksPerCopy, B, n1, n2);
  }));
  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "sm_block_trans_cuda_dx", ([&] {
    sm_block_trans_cuda_dx_kernel<scalar_t><<<dx_blocks, threads>>>(
        grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        image.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        B, N1, N2);
  }));

  return {grad_x, grad_w, grad_b};
}