#include <torch/extension.h>

#include <vector>
#include "sm_block_cpu.h"

// CUDA forward declarations
std::vector<torch::Tensor> sm_block_cuda_forward(
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias);

std::vector<torch::Tensor> sm_block_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> sm_block_forward(
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {
  CHECK_CONTIGUOUS(image);
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(weights);
  CHECK_CONTIGUOUS(bias);
  if (image.device().is_cuda() && x.device().is_cuda() && weights.device().is_cuda() && bias.device().is_cuda())
    return sm_block_cuda_forward(image, x, weights, bias);
  else if (!image.device().is_cuda() && !x.device().is_cuda() && !weights.device().is_cuda() && !bias.device().is_cuda())
    return sm_block_cpu_forward(image, x, weights, bias);
  std::cout << "All tensors must be CPU or CUDA" << std::endl;
}

std::vector<torch::Tensor> sm_block_backward(
    torch::Tensor grad_output,
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {
  CHECK_CONTIGUOUS(grad_output);
  CHECK_CONTIGUOUS(image);
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(weights);
  CHECK_CONTIGUOUS(bias);
  if (grad_output.device().is_cuda() && image.device().is_cuda() && x.device().is_cuda() && weights.device().is_cuda() && bias.device().is_cuda())
    return sm_block_cuda_backward(grad_output, image, x, weights, bias);
  else if (!grad_output.device().is_cuda() && !image.device().is_cuda() && !x.device().is_cuda() && !weights.device().is_cuda() && !bias.device().is_cuda())
    return sm_block_cpu_backward(grad_output, image, x, weights, bias);
  std::cout << "All tensors must be CPU or CUDA" << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sm_block_forward, "SM block forward");
  m.def("backward", &sm_block_backward, "SM block backward");
}
