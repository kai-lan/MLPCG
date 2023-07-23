#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define NUM_IMAGES 3
#define KERNEL_SIZE 27
#define WEIGHT_SIZE NUM_IMAGES*KERNEL_SIZE*KERNEL_SIZE
__constant__ unsigned char WEIGHT_BYTES[WEIGHT_SIZE*sizeof(double)];

#define LOCATIONSPERBLOCK 4
#define NUM_THREADS_FORWARD 256
#define NUM_THREADS_BACKWARD 256

namespace {

template <size_t blockSize, typename T>
__device__ void warpReduce(volatile T *sdata, size_t tid)
{
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32]; __syncthreads();
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16]; __syncthreads();
    if (blockSize >= 16) sdata[tid] += sdata[tid +  8]; __syncthreads();
    if (blockSize >=  8) sdata[tid] += sdata[tid +  4]; __syncthreads();
    if (blockSize >=  4) sdata[tid] += sdata[tid +  2]; __syncthreads();
    if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
}

template <typename scalar_t>
__global__ void sm_linear_3d_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image, // 1, N, N, N
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> y,
    const int nBlocks, const int N1, const int N2, const int N3) {

  __shared__ scalar_t z[NUM_THREADS_FORWARD];

  const scalar_t *WEIGHT = (const scalar_t *)(WEIGHT_BYTES);

  const int outerBlock = blockIdx.x / nBlocks;
  const int innerBlock = blockIdx.x % nBlocks;
  const int c = outerBlock;
  const int tid = threadIdx.x;
  const int location = blockDim.x * innerBlock + tid;

  scalar_t zz = 0.0;
  if (location < NUM_IMAGES*N1*N2*N3) {
    const int ij = location % N3;
    const int j = (location/N3) % N2;
    const int i = (location/N3) / N2;

    int c0, c1, c2;
    int i1 = i + 1, i2 = i + 2;
    int j1 = j + 1, j2 = j + 2;
    int ij1 = ij + 1, ij2 = ij + 2;
    #pragma unroll(1)
    for (int m = 0; m < NUM_IMAGES; ++m) {
      c0 = NUM_IMAGES*c+m;
          c1 = 27 * c0;
          zz += WEIGHT[c1] * image[m][i][j][ij]; // 27 * c0
          c1 ++;
          zz += WEIGHT[c1] * image[m][i][j][ij1]; // 27 * c0 + 1
          c1 ++;
          zz += WEIGHT[c1] * image[m][i][j][ij2]; // 27 * c0 + 2

          c1 ++;
          zz += WEIGHT[c1] * image[m][i][j1][ij]; // 3 * (3 * 3 * c0 + 1) = 27 * c0 + 3
          c1 ++;
          zz += WEIGHT[c1] * image[m][i][j1][ij1]; // 27 * c0 + 4
          c1 ++;
          zz += WEIGHT[c1] * image[m][i][j1][ij2]; // 27 * c0 + 5

          c1 ++;
          zz += WEIGHT[c1] * image[m][i][j2][ij]; // 3 * (3 * (3 * c0) + 2) = 27 * c0 + 6
          c1 ++;
          zz += WEIGHT[c1] * image[m][i][j2][ij1]; // 27 * c0 + 7
          c1 ++;
          zz += WEIGHT[c1] * image[m][i][j2][ij2]; // 27 * c0 + 8


          c1 ++;
          zz += WEIGHT[c1] * image[m][i1][j][ij]; // 3 * (3 * (3 * c0 + 1)) = 27 * c0 + 9
          c1 ++;
          zz += WEIGHT[c1] * image[m][i1][j][ij1];
          c1 ++;
          zz += WEIGHT[c1] * image[m][i1][j][ij2];

          c1 ++;
          zz += WEIGHT[c1] * image[m][i1][j1][ij];
          c1 ++;
          zz += WEIGHT[c1] * image[m][i1][j1][ij1];
          c1 ++;
          zz += WEIGHT[c1] * image[m][i1][j1][ij2];

          c1 ++;
          zz += WEIGHT[c1] * image[m][i1][j2][ij];
          c1 ++;
          zz += WEIGHT[c1] * image[m][i1][j2][ij1];
          c1 ++;
          zz += WEIGHT[c1] * image[m][i1][j2][ij2];


          c1 ++;
          zz += WEIGHT[c1] * image[m][i2][j][ij];
          c1 ++;
          zz += WEIGHT[c1] * image[m][i2][j][ij1];
          c1 ++;
          zz += WEIGHT[c1] * image[m][i2][j][ij2];

          c1 ++;
          zz += WEIGHT[c1] * image[m][i2][j1][ij];
          c1 ++;
          zz += WEIGHT[c1] * image[m][i2][j1][ij1];
          c1 ++;
          zz += WEIGHT[c1] * image[m][i2][j1][ij2];

          c1 ++;
          zz += WEIGHT[c1] * image[m][i2][j2][ij];
          c1 ++;
          zz += WEIGHT[c1] * image[m][i2][j2][ij1];
          c1 ++;
          zz += WEIGHT[c1] * image[m][i2][j2][ij2];

    }
  }
  z[tid] = zz;
  __syncthreads();

  // reduction
  if (NUM_THREADS_FORWARD >= 1024) { if (tid < 512) { z[tid] += z[tid + 512]; } __syncthreads(); }
  if (NUM_THREADS_FORWARD >=  512) { if (tid < 256) { z[tid] += z[tid + 256]; } __syncthreads(); }
  if (NUM_THREADS_FORWARD >=  256) { if (tid < 128) { z[tid] += z[tid + 128]; } __syncthreads(); }
  if (NUM_THREADS_FORWARD >=  128) { if (tid <  64) { z[tid] += z[tid +  64]; } __syncthreads(); }

  if (tid < 32) warpReduce<NUM_THREADS_FORWARD, scalar_t>(z, tid);
  if (tid == 0) atomicAdd(&y[0], z[0]);
}


template <typename scalar_t>
__global__ void sm_linear_3d_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> grad_output, // bs, 1, N, N, N
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> image, // 1, N, N, N
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_w,
    const int nBlocksPerCopy, const int n1, const int n2, const int n3) {

  __shared__ scalar_t I[LOCATIONSPERBLOCK+2][LOCATIONSPERBLOCK+2][LOCATIONSPERBLOCK+2];

  const int location = blockIdx.x / nBlocksPerCopy;
  const int innerBlock = blockIdx.x % nBlocksPerCopy;

  const int ij = (location % n3) * LOCATIONSPERBLOCK;
  const int j = ((location/n3) % n2) * LOCATIONSPERBLOCK;
  const int i = ((location/n3) / n2) * LOCATIONSPERBLOCK;

  int p = innerBlock * blockDim.x + threadIdx.x;
  const int kl = p % 3;
  p /= 3;
  const int l = p % 3;
  p /= 3;
  const int k = p % 3;
  p /= 3;
  const int m = p % NUM_IMAGES;
  const int c = p / NUM_IMAGES;

  int tid = threadIdx.x;
  while (tid < (LOCATIONSPERBLOCK+2)*(LOCATIONSPERBLOCK+2)*(LOCATIONSPERBLOCK+2)) {
    int iz = tid % (LOCATIONSPERBLOCK+2);
    int iy = (tid/(LOCATIONSPERBLOCK+2)) % (LOCATIONSPERBLOCK+2);
    int ix = (tid/(LOCATIONSPERBLOCK+2)) / (LOCATIONSPERBLOCK+2);
    I[ix][iy][iz] = image[m][i+ix][j+iy][ij+iz];
    tid += blockDim.x;
  }

  if (c >= KERNEL_SIZE) return; // Wasted threads
  __syncthreads();

  scalar_t d_w = 0.0;
  #pragma unroll(1)
  for (int _i = 0; _i < LOCATIONSPERBLOCK; ++_i) {
    for (int _j = 0; _j < LOCATIONSPERBLOCK; ++_j) {
      for (int _ij = 0; _ij < LOCATIONSPERBLOCK; ++_ij) {
        d_w += I[_i+k][_j+l][_ij+kl];
        // d_w += image[m][i+_i+k][j+_j+l][ij+_ij+kl];
      }
    }
  }

  atomicAdd(&grad_w[c][m][k][l][kl], d_w);
}

} // namespace

std::vector<torch::Tensor> sm_linear_3d_cuda_forward(
    torch::Tensor image,
    torch::Tensor weights,
    torch::Tensor bias) {

  assert(image.size(0) == NUM_IMAGES);

  if (image.dtype() == torch::ScalarType::Double) {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<double>(), WEIGHT_SIZE * sizeof(double));
  } else {
    cudaMemcpyToSymbol(WEIGHT_BYTES, weights.data_ptr<float>(), WEIGHT_SIZE * sizeof(float));
  }

  const int N1 = image.size(1)-2;
  const int N2 = image.size(2)-2;
  const int N3 = image.size(3)-2;
  auto y = torch::zeros({1}, torch::dtype(image.dtype()).device(image.device()));

  const int nThreads = NUM_THREADS_FORWARD;
  const int nBlocks = (N1*N2*N3 + nThreads - 1) / nThreads; // b, i, j, ij

  const dim3 threads(nThreads);
  const dim3 blocks(nBlocks * KERNEL_SIZE);

  AT_DISPATCH_FLOATING_TYPES(image.type(), "sm_linear_3d_forward_cuda", ([&] {
    sm_linear_3d_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        y.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        nBlocks, N1, N2, N3);
  }));

  y /= KERNEL_SIZE * N1*N2*N3;
  y += bias.mean();
  return {y};
}

std::vector<torch::Tensor> sm_linear_3d_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor image) {

  assert(image.size(0) == NUM_IMAGES);

  const int N1 = image.size(1) - 2;
  const int N2 = image.size(2) - 2;
  const int N3 = image.size(3) - 2;

  const int nThreads = NUM_THREADS_BACKWARD;
  const int nBlocksPerCopy = (WEIGHT_SIZE + nThreads - 1) / nThreads;
  const int locationsPerBlock = LOCATIONSPERBLOCK;

  assert((N1 % locationsPerBlock == 0) && (N2 % locationsPerBlock == 0) && (N3 % locationsPerBlock == 0)); // Data must be divisible by divisions

  const int n1 = N1 / locationsPerBlock;
  const int n2 = N2 / locationsPerBlock;
  const int n3 = N3 / locationsPerBlock;
  const dim3 threads(nThreads);
  const dim3 blocks(nBlocksPerCopy*n1*n2*n3);

  auto grad_w = torch::zeros({27, NUM_IMAGES, 3, 3, 3}, torch::dtype(image.dtype()).device(image.device()));
  auto grad_b = torch::ones({27}, torch::dtype(image.dtype()).device(image.device())) / KERNEL_SIZE * grad_output;

  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "sm_linear_3d_cuda_backward", ([&] {
    sm_linear_3d_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        image.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_w.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        nBlocksPerCopy, n1, n2, n3);
  }));
  grad_w /= KERNEL_SIZE * N1 * N2 * N3;
  grad_w *= grad_output;
  return {grad_w, grad_b};
}

int main() {
  int bs = 5;
  int N = 128;
  int num_imgs = 3;
  auto image = torch::ones({num_imgs, N, N, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto padded_image = torch::ones({num_imgs, N+2, N+2, N+2}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto x = torch::rand({bs, 1, N, N, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto weight = torch::rand({27, num_imgs, 3, 3, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto bias = torch::rand({27}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto grad_output = torch::ones({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto y = torch::zeros({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  // curandState *d_states;
  // cudaMalloc(&d_states, CNT * sizeof(curandState));
  // kernel_setup_randstates_2d<<<1,CNT>>>(d_states, 1,1, 1);
  // cudaDeviceSynchronize();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < 100; ++i)
    y = sm_linear_3d_cuda_backward(grad_output, padded_image)[0];
    // y = sm_linear_3d_cuda_forward(padded_image, weight, bias)[0];

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "time " << milliseconds/100*1000 << " us" << std::endl;
  // cudaDeviceSynchronize();
}