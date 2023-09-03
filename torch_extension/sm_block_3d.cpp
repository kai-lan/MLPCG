#include <torch/extension.h>

#include <vector>


std::vector<torch::Tensor> sm_block_3d_cuda_inference(
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias);

std::vector<torch::Tensor> sm_block_3d_cuda_forward(
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias);

std::vector<torch::Tensor> sm_block_3d_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> sm_block_3d_inference(
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {
  CHECK_INPUT(image);
  CHECK_INPUT(x);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);

  return sm_block_3d_cuda_inference(image, x, weights, bias);
}

std::vector<torch::Tensor> sm_block_3d_forward(
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {
  CHECK_INPUT(image);
  CHECK_INPUT(x);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);

  return sm_block_3d_cuda_forward(image, x, weights, bias);
}

std::vector<torch::Tensor> sm_block_3d_backward(
    torch::Tensor grad_output,
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(image);
  CHECK_INPUT(x);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);

  return sm_block_3d_cuda_backward(grad_output, image, x, weights, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inference", &sm_block_3d_inference, "SM block 3D inference (CUDA)");
  m.def("forward", &sm_block_3d_forward, "SM block 3D forward (CUDA)");
  m.def("backward", &sm_block_3d_backward, "SM block 3D backward (CUDA)");
}
