#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

void gemm(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C,
          float alpha = 1.0f, float beta = 0.0f) {
  // 获取输入张量的维度和大小
  const int64_t m = A.size(0);
  const int64_t k = A.size(1);
  const int64_t n = B.size(1);

  // 获取张量的指针
  const float *A_ptr = A.data_ptr<float>();
  const float *B_ptr = B.data_ptr<float>();
  float *C_ptr = C.data_ptr<float>();

  // 创建CUBLAS句柄
  cublasHandle_t handle;
  cublasCreate(&handle);

  // 指定矩阵大小和alpha/beta值
  const float *alpha_ptr = &alpha;
  const float *beta_ptr = &beta;

  // 执行GEMM操作
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha_ptr, B_ptr, n,
              A_ptr, k, beta_ptr, C_ptr, n);

  // 销毁CUBLAS句柄
  cublasDestroy(handle);
}

int main() {
  // 创建输入张量
  auto A = torch::ones({2, 3}).cuda();
  auto B = torch::ones({3, 4}).cuda();

  // 创建输出张量
  auto C = torch::zeros({2, 4}).cuda();

  // 执行GEMM操作
  gemm(A, B, C);

  // 输出结果张量
  std::cout << C << std::endl;

  return 0;
}