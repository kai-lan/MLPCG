#include <torch/extension.h>

#include <vector>
#include "sm_linear_cpu.h"

std::vector<torch::Tensor> sm_linear_cuda_forward(
    torch::Tensor image,
    torch::Tensor weights,
    torch::Tensor bias);

std::vector<torch::Tensor> sm_linear_cuda_inference(
    torch::Tensor image,
    torch::Tensor weights,
    torch::Tensor bias);

std::vector<torch::Tensor> sm_linear_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor y);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> sm_linear_forward(
    torch::Tensor image,
    torch::Tensor weights,
    torch::Tensor bias) {
  CHECK_CONTIGUOUS(image);
  CHECK_CONTIGUOUS(weights);
  CHECK_CONTIGUOUS(bias);
  if (image.device().is_cuda() && weights.device().is_cuda() && bias.device().is_cuda())
    return sm_linear_cuda_forward(image, weights, bias);
  else if (!image.device().is_cuda() && !weights.device().is_cuda() && !bias.device().is_cuda())
    return sm_linear_cpu_forward(image, weights, bias);
  std::cout << "All tensors must be CPU or CUDA" << std::endl;
}

std::vector<torch::Tensor> sm_linear_inference(
    torch::Tensor image,
    torch::Tensor weights,
    torch::Tensor bias) {
  CHECK_INPUT(image);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);

  return sm_linear_cuda_inference(image, weights, bias);
}

std::vector<torch::Tensor> sm_linear_backward(
    torch::Tensor grad_output,
    torch::Tensor y) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(y);
  if (grad_output.device().is_cuda() && y.device().is_cuda())
    return sm_linear_cuda_backward(grad_output, y);
  else if (!grad_output.device().is_cuda() && !y.device().is_cuda())
    return sm_linear_cuda_backward(grad_output, y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sm_linear_forward, "SM linear forward");
  m.def("inference", &sm_linear_inference, "SM linear inference");
  m.def("backward", &sm_linear_backward, "SM linear backward");
}
