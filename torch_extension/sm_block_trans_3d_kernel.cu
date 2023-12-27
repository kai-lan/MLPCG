#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>    // This contains gpuAtomicAdd
#include <vector>

#define NUM_IMAGES 3
#define KERNEL_SIZE 27
#define WEIGHT_SIZE NUM_IMAGES*KERNEL_SIZE*KERNEL_SIZE
__constant__ unsigned char WEIGHT_BYTES[WEIGHT_SIZE*sizeof(double)];
__constant__ unsigned char BIAS_BYTES[KERNEL_SIZE*sizeof(double)];

#define LOCATIONS_PER_BLOCK 8
#define NUM_THREADS_INFERENCE 256 // 1024/512 produced wrong results, don't know why
#define NUM_THREADS_FORWARD 256
#define NUM_THREADS_BACKWARD 256

namespace {
template <typename scalar_t>
__global__ void sm_block_trans_3d_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image, // num_imgs, N, N
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> x, // bs, 1, N, N
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

  for (int k = 0; k <= 2; ++k) {
    int ii = i + k - 1;
    if (ii < 0 || ii >= N1) continue;
    for (int l = 0; l <= 2; ++l) {
      int jj = j + l - 1;
      if (jj < 0 || jj >= N2) continue;
      for (int kl = 0; kl <= 2; ++kl) {
        int iijj = ij + kl - 1;
        if (iijj < 0 || iijj >= N3) continue;
        int p = 3 * (3 * k + l) + kl;
        scalar_t K_ij = BIAS[p];
        for (int s = 0; s < NUM_IMAGES; ++s) {
          for (int m = 0; m <= 2; ++m) {
            int aa = i+m;
            for (int n = 0; n <= 2; ++n) {
              int bb = j+n;
              for (int mn = 0; mn <=2; ++mn) {
                int cc = ij+mn;
                K_ij += WEIGHT[3*(3*(3*(NUM_IMAGES*p+s)+m)+n)+mn] * image[s][aa][bb][cc];
              }
            }
          }
        }

        scalar_t val = K_ij * x[c][0][i+1][j+1][ij+1];
        atomicAdd(&y[c][0][ii][jj][iijj], val);
      }
    }
  }
} // forward kernel

template <typename scalar_t>
__global__ void sm_block_trans_3d_cuda_dwdb_fast_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_output, // bs, 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image, // 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> x, // bs, 1, N, N, N
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_w,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> grad_b,
    const int nBlocksPerCopy, const int B, const int n1, const int n2, const int n3,
    const int locations_per_block) {

  __shared__ scalar_t OUT[LOCATIONS_PER_BLOCK+2][LOCATIONS_PER_BLOCK+2][LOCATIONS_PER_BLOCK+2]; // 6^3 = 216
  __shared__ scalar_t I[NUM_IMAGES][LOCATIONS_PER_BLOCK+2][LOCATIONS_PER_BLOCK+2][LOCATIONS_PER_BLOCK+2]; // 3 * 6^3 = 648
  __shared__ scalar_t X[LOCATIONS_PER_BLOCK][LOCATIONS_PER_BLOCK][LOCATIONS_PER_BLOCK]; // 4^3 = 64

  const int N1 = locations_per_block * n1;
  const int N2 = locations_per_block * n2;
  const int N3 = locations_per_block * n3;

  int location = blockIdx.x / nBlocksPerCopy;
  const int innerBlock = blockIdx.x % nBlocksPerCopy;

  const int ij = (location % n3) * locations_per_block;
  location /= n3;
  const int j = (location % n2) * locations_per_block;
  location /= n2;
  const int i = (location % n1) * locations_per_block;
  const int c = location / n1;

  const int width = locations_per_block+2;
  const int OUT_Size = width*width*width; // 216
  const int X_Size = locations_per_block*locations_per_block*locations_per_block; // 64
  const int I_Size = NUM_IMAGES*width*width*width; // 648

  int tid = threadIdx.x;
  while (tid < I_Size + X_Size + OUT_Size) {
    int ttid = tid;
    if (ttid < I_Size) {
      int iz = tid % width;
      tid /= width;
      int iy = tid % width;
      tid /= width;
      int ix = tid % width;
      int s = tid / width;
      I[s][ix][iy][iz] = image[s][i+ix][j+iy][ij+iz];
    }
    else if (ttid < I_Size + OUT_Size) {
      tid -= I_Size;
      int iz = tid % width;
      tid /= width;
      int iy = tid % width;
      int ix = tid / width;
      int ii = i + ix - 1, jj = j + iy - 1, iijj = ij + iz - 1;
      if (ii < 0 || ii >= N1 || jj < 0 || jj >= N2 || iijj < 0 || iijj >= N3)
        OUT[ix][iy][iz] = 0.0;
      else
        OUT[ix][iy][iz] = grad_output[c][0][ii][jj][iijj];
    }
    else {
      tid -= I_Size + OUT_Size;
      int iz = tid % locations_per_block;
      tid /= locations_per_block;
      int iy = tid % locations_per_block;
      int ix = tid / locations_per_block;
      X[ix][iy][iz] = x[c][0][i+ix+1][j+iy+1][ij+iz+1];
    }

    tid = ttid + blockDim.x;
  }


  const int p = innerBlock * blockDim.x + threadIdx.x;
  if (p >= WEIGHT_SIZE + KERNEL_SIZE) return; // Wasted threads

  const int idx_kl = p / (NUM_IMAGES*KERNEL_SIZE + 1);
  const int kl = idx_kl % 3;
  const int l = (idx_kl / 3) % 3;
  const int k = idx_kl / 9;

  __syncthreads();

  const int idx_smn = p % (NUM_IMAGES*KERNEL_SIZE + 1);

  int s, mn, n, m;
  int idx;

  scalar_t d_w = 0.0, d_b = 0.0;
  if (idx_smn == NUM_IMAGES * KERNEL_SIZE) {
    #pragma unroll
    for (int _i = 0; _i < locations_per_block; ++_i) {
      for (int _j = 0; _j < locations_per_block; ++_j) {
        for (int _ij = 0; _ij < locations_per_block; ++_ij) {
          d_b += OUT[_i+k][_j+l][_ij+kl] * X[_i][_j][_ij];
        }
      }
    }
  } else {
    mn = idx_smn % 3;
    n = (idx_smn / 3) % 3;
    m = (idx_smn / 9) % 3;
    s = idx_smn / 27;
    idx = KERNEL_SIZE*NUM_IMAGES*idx_kl + idx_smn;

    #pragma unroll
    for (int _i = 0; _i < locations_per_block; ++_i) {
      for (int _j = 0; _j < locations_per_block; ++_j) {
        for (int _ij = 0; _ij < locations_per_block; ++_ij) {
          d_w += OUT[_i+k][_j+l][_ij+kl] * I[s][_i+m][_j+n][_ij+mn] * X[_i][_j][_ij];
        }
      }
    }
  }

  if (idx_smn == NUM_IMAGES * KERNEL_SIZE)
    atomicAdd(&grad_b[idx_kl], d_b);
  else
    atomicAdd(&grad_w[idx_kl][s][m][n][mn], d_w);
}


template <typename scalar_t>
__global__ void sm_block_trans_3d_cuda_dx_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_output, // bs, 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image, // 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> x, // bs, 1, N, N, N
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_x,
    const int B, const int N1, const int N2, const int N3) {

  const scalar_t *WEIGHT = (const scalar_t *)(WEIGHT_BYTES);
  const scalar_t *BIAS = (const scalar_t *)(BIAS_BYTES);

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId < B * N1 * N2 * N3) {
    const int ij = threadId % N3;
    threadId /= N3;
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
        for (int kl = 0; kl <= 2; ++kl) {
          int iijj = ij + kl - 1;
          if (iijj < 0 || iijj >= N3) continue;

          int idx_kl = 3 * (3 * k + l) + kl;

          scalar_t z = BIAS[idx_kl];
          #pragma unroll
          for (int s = 0; s < NUM_IMAGES; ++s) {
            for (int m = 0; m <= 2; ++m) {
              for (int n = 0; n <= 2; ++n) {
                for (int mn = 0; mn <= 2; ++mn) {
                  z += WEIGHT[3*(3*(3*(NUM_IMAGES*idx_kl+s)+m)+n)+mn] * image[s][i+m][j+n][ij+mn];
                }
              }
            }
          }
          z *= grad_output[c][0][ii][jj][iijj];
          zz += z;
        }
      }
    }
    grad_x[c][0][i][j][ij] = zz;
  }
} // backward dx
} // namespace

std::vector<torch::Tensor> sm_block_trans_3d_cuda_inference(
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {
  assert(image.size(0) == NUM_IMAGES);

  if (x.dtype() == torch::ScalarType::Double) {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<double>(), WEIGHT_SIZE * sizeof(double));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<double>(), KERNEL_SIZE * sizeof(double));
  }
  else if (x.dtype() == torch::ScalarType::Float){
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<float>(), WEIGHT_SIZE * sizeof(float));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<float>(), KERNEL_SIZE * sizeof(float));
  }
  else {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<at::Half>(), WEIGHT_SIZE * sizeof(at::Half));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<at::Half>(), KERNEL_SIZE * sizeof(at::Half));
  }

  const int B = x.size(0);
  const int N1 = image.size(1)-2;
  const int N2 = image.size(2)-2;
  const int N3 = image.size(3)-2;
  auto y = torch::zeros({B, x.size(1), N1, N2, N3}, torch::dtype(x.dtype()).device(x.device()));

  const int nThreads = NUM_THREADS_INFERENCE;
  const int nBlocks = (B*N1*N2*N3 + nThreads - 1) / nThreads;

  const dim3 threads(nThreads);
  const dim3 blocks(nBlocks);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), "sm_block_trans_3d_inference_cuda", ([&] {
    sm_block_trans_3d_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        y.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        B, N1, N2, N3);
  }));

  return {y};
}

std::vector<torch::Tensor> sm_block_trans_3d_cuda_forward(
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {
  assert(image.size(0) == NUM_IMAGES);

  if (x.dtype() == torch::ScalarType::Double) {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<double>(), WEIGHT_SIZE * sizeof(double));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<double>(), KERNEL_SIZE * sizeof(double));
  }
  else if (x.dtype() == torch::ScalarType::Float){
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<float>(), WEIGHT_SIZE * sizeof(float));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<float>(), KERNEL_SIZE * sizeof(float));
  }
  else {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<at::Half>(), WEIGHT_SIZE * sizeof(at::Half));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<at::Half>(), KERNEL_SIZE * sizeof(at::Half));
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

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), "sm_block_trans_3d_forward_cuda", ([&] {
    sm_block_trans_3d_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        y.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        B, N1, N2, N3);
  }));

  return {y};
}

std::vector<torch::Tensor> sm_block_trans_3d_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {

  assert(image.size(0) == NUM_IMAGES);

  if (x.dtype() == torch::ScalarType::Double) {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<double>(), WEIGHT_SIZE * sizeof(double));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<double>(), KERNEL_SIZE * sizeof(double));
  }
  else if (x.dtype() == torch::ScalarType::Float){
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<float>(), WEIGHT_SIZE * sizeof(float));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<float>(), KERNEL_SIZE * sizeof(float));
  }
  else {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<at::Half>(), WEIGHT_SIZE * sizeof(at::Half));
    cudaMemcpyToSymbol(BIAS_BYTES, bias.data_ptr<at::Half>(), KERNEL_SIZE * sizeof(at::Half));
  }

  const int B = x.size(0);
  const int N1 = image.size(1)-2;
  const int N2 = image.size(2)-2;
  const int N3 = image.size(3)-2;
  const int totalData = B * N1 * N2 * N3;

  const int nThreads = NUM_THREADS_BACKWARD;
  const int nBlocksPerCopy = (WEIGHT_SIZE + KERNEL_SIZE + nThreads - 1) / nThreads;
  const int locationsPerBlock = std::min({LOCATIONS_PER_BLOCK, N1, N2, N3});

  assert((N1 % locationsPerBlock == 0) && (N2 % locationsPerBlock == 0) && (N3 % locationsPerBlock == 0)); // Data must be divisible by divisions

  const int n1 = N1 / locationsPerBlock;
  const int n2 = N2 / locationsPerBlock;
  const int n3 = N3 / locationsPerBlock;
  const dim3 threads(nThreads);
  const dim3 dwdb_blocks(nBlocksPerCopy*(B*n1*n2*n3)), dx_blocks((totalData + nThreads - 1) / nThreads);

  auto grad_x = torch::zeros({B, x.size(1), N1, N2, N3}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_w = torch::zeros({27, NUM_IMAGES, 3, 3, 3}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_b = torch::zeros({27}, torch::dtype(x.dtype()).device(x.device()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.type(), "sm_block_trans_3d_cuda_dwdb", ([&] {
    sm_block_trans_3d_cuda_dwdb_fast_kernel<scalar_t><<<dwdb_blocks, threads>>>(
        grad_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_w.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_b.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        nBlocksPerCopy, B, n1, n2, n3,
        locationsPerBlock);
  }));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.type(), "sm_block_trans_3d_cuda_dx", ([&] {
    sm_block_trans_3d_cuda_dx_kernel<scalar_t><<<dx_blocks, threads>>>(
        grad_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        B, N1, N2, N3);
  }));

  return {grad_x, grad_w, grad_b};
}