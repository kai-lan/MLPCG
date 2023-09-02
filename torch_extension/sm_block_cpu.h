#include <torch/extension.h>
#include <vector>

#define NUM_IMAGES 3
#define KERNEL_SIZE 9
#define WEIGHT_SIZE NUM_IMAGES*KERNEL_SIZE*KERNEL_SIZE


std::vector<torch::Tensor> sm_block_cpu_forward(
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {

  assert(image.size(0) == NUM_IMAGES);
  const int B = x.size(0);
  const int N1 = image.size(1)-2;
  const int N2 = image.size(2)-2;
  auto y = torch::zeros({B, x.size(1), N1, N2}, torch::dtype(x.dtype()).device(x.device()));

  #pragma omp parallel for
  for (int c = 0; c < B; ++c) {
    for (int i = 0; i < N1; ++i) {
      for (int j = 0; j < N2; ++j) {
        for (int k = 0; k <= 2; ++k) {
          for (int l = 0; l <= 2; ++l) {
            int p = 3 * k + l;
            auto z = bias[p];
            for (int s = 0; s < NUM_IMAGES; ++s) {
              for (int m = 0; m <= 2; ++m) {
                for (int n = 0; n <= 2; ++n) {
                  z = z + weights[p][s][m][n] * image[s][i+m][j+n];
                }
              }
            }
            z = z * x[c][0][i+k][j+l];
            y[c][0][i][j] += z;
          }
        }
      }
    }
  }

  return {y};
}

// Not implemented
std::vector<torch::Tensor> sm_block_cpu_backward(
    torch::Tensor grad_output,
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {

  assert(image.size(0) == NUM_IMAGES);
  const int B = x.size(0);
  const int N1 = image.size(1)-2;
  const int N2 = image.size(2)-2;

  auto grad_x = torch::zeros({B, x.size(1), N1, N2}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_w = torch::zeros({9, NUM_IMAGES, 3, 3}, torch::dtype(x.dtype()).device(x.device()));
  auto grad_b = torch::zeros({9}, torch::dtype(x.dtype()).device(x.device()));

  return {grad_x, grad_w, grad_b};
}