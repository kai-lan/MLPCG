#include <torch/extension.h>
#include <vector>

#define NUM_IMAGES 3
#define KERNEL_SIZE 9

std::vector<torch::Tensor> sm_linear_cpu_forward(
  torch::Tensor image,
  torch::Tensor weights,
  torch::Tensor bias) {
  std::cout << "SM linear CPU forward" << std::endl;
  assert(image.size(0) == NUM_IMAGES);

  const int N1 = image.size(1)-2;
  const int N2 = image.size(2)-2;

  auto y = torch::zeros({9, NUM_IMAGES, 3, 3}, torch::dtype(image.dtype()).device(image.device())); // useful for backward
  auto y_sum = torch::zeros({1}, torch::dtype(image.dtype()).device(image.device()));

  #pragma omp parallel for
  for (int i = 0; i < N1; ++i) {
    for (int j = 0; j < N2; ++j) {
      for (int c = 0; c < KERNEL_SIZE; ++c) {
        for (int m = 0; m < NUM_IMAGES; ++m) {
          for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
              y[c][m][k][l] += image[m][i+k][j+l];
            }
          }
        }
      }
    }
  }

  y /= KERNEL_SIZE * N1*N2;
  y_sum[0] = weights.ravel().dot(y.ravel());
  y_sum += bias.mean();
  return {y_sum, y};
}

std::vector<torch::Tensor> sm_linear_cpu_backward(
  torch::Tensor grad_output,
  torch::Tensor y) {

  auto grad_b = torch::ones({9}, torch::dtype(grad_output.dtype()).device(grad_output.device())) / KERNEL_SIZE * grad_output;

  auto grad_w = y * grad_output;

  return {grad_w, grad_b};
}