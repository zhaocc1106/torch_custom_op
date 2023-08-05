// my_add.cpp
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

torch::Tensor my_matmul2(torch::Tensor a, torch::Tensor b) {
  auto a_cuda = a.cuda();
  auto b_cuda = b.cuda();

  // 获取输入张量的维度和大小
  const int64_t m = a.size(0);
  const int64_t k = a.size(1);
  const int64_t n = b.size(1);

  // 获取张量的指针
  const float *a_ptr = a_cuda.data_ptr<float>();
  const float *b_ptr = b_cuda.data_ptr<float>();
  auto c = torch::zeros({m, n}).cuda();
  float *c_ptr = c.data_ptr<float>();

  // 创建CUBLAS句柄
  cublasHandle_t handle;
  cublasCreate(&handle);

  // 指定矩阵大小和alpha/beta值
  float alpha = 1.0f;
  float beta = 0.0f;
  const float *alpha_ptr = &alpha;
  const float *beta_ptr = &beta;

  // 执行GEMM操作
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha_ptr, b_ptr, n,
              a_ptr, k, beta_ptr, c_ptr, n);

  // 销毁CUBLAS句柄
  cublasDestroy(handle);

  if (a.device().is_cpu()) {
    return c.cpu();
  } else {
    return c;
  }
}

PYBIND11_MODULE(my_lib2, m) {
  m.def("my_matmul", my_matmul2);
}