#include <torch/extension.h>

#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> sm_block_trans_cuda_forward(
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> sm_block_trans_forward(
    torch::Tensor image,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor bias) {
  CHECK_INPUT(image);
  CHECK_INPUT(x);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);

  return sm_block_trans_cuda_forward(image, x, weights, bias);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sm_block_trans_forward, "SM block transpose forward");
}
