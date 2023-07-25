#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> sm_linear_3d_cuda_forward(
    torch::Tensor image,
    torch::Tensor weights,
    torch::Tensor bias);

std::vector<torch::Tensor> sm_linear_3d_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor y);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> sm_linear_3d_forward(
    torch::Tensor image,
    torch::Tensor weights,
    torch::Tensor bias) {
  CHECK_INPUT(image);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);

  return sm_linear_3d_cuda_forward(image, weights, bias);
}

std::vector<torch::Tensor> sm_linear_3d_backward(
    torch::Tensor grad_output,
    torch::Tensor y) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(y);

  return sm_linear_3d_cuda_backward(grad_output, y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sm_linear_3d_forward, "SM linear 3D forward (CUDA)");
  m.def("backward", &sm_linear_3d_backward, "SM linear 3D backward (CUDA)");
}
