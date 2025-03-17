#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

// Use this macro to check the cuda_error code
#define CHECK_CUDA(call)                                                   \
  {                                                                        \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), \
              __FILE__, __LINE__);                                         \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  }

// Macro to check the cusparseStatus
#define CHECK_CUSPARSE(call)                                            \
  {                                                                     \
    cusparseStatus_t status = call;                                     \
    if (status != CUSPARSE_STATUS_SUCCESS) {                            \
      fprintf(stderr, "cuSPARSE error: %d (%s:%d)\n", status, __FILE__, \
              __LINE__);                                                \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

int read_mtx(const std::string mtxpath, std::vector<int> &row_ptr,
             std::vector<int> &col_idx, std::vector<float> &vals) {
  std::ifstream fin(mtxpath);
  if (!fin.is_open()) {
    std::cerr << "Cannot open file " << mtxpath << std::endl;
    return -1;
  }
  while (fin.peek() == '%') fin.ignore(2048, '\n');

  int M, N, L;
  fin >> M >> N >> L;
  col_idx.resize(L);
  vals.resize(L);
  row_ptr.assign(M + 1, 0);
  int prev_row = -1;
  for (int l = 0; l < L; l++) {
    int m, n;
    float data;
    fin >> m >> n >> data;
    int row = m - 1;
    int col = n - 1;
    col_idx[l] = col;
    vals[l] = data;
    if (row != prev_row) {
      for (int r = prev_row + 1; r <= row; r++) {
        row_ptr[r] = l;
      }
      prev_row = row;
    }
  }
  for (int r = prev_row + 1; r <= M; r++) {
    row_ptr[r] = L;
  }
  fin.close();
  return N;
}

int main(int argc, char *argv[]) {
  // STEP 1: Read Matrix from mtx file
  if (argc < 2) {
    std::cerr << "Passed arguments: " << argc << std::endl;
    std::cerr << "Usage: " << argv[0] << " <matrixpath>" << std::endl;
    return 1;
  }
  std::vector<int> h_csrRowPtr;
  std::vector<int> h_csrColInd;
  std::vector<float> h_csrVal;
  std::string matrixfile = static_cast<std::string>(argv[1]);
  int N = read_mtx(matrixfile, h_csrRowPtr, h_csrColInd, h_csrVal);

  srand(time(NULL));
  int M = h_csrRowPtr.size() - 1;
  int nnz = h_csrVal.size();
  std::cout << "Matrix Read" << std::endl
            << "Number of Rows: " << M << std::endl
            << "Number of Cols: " << N << std::endl
            << "Number of NNZ: " << nnz << std::endl;

  // STEP 2: Create the Random Vector for Y= A*x
  std::vector<float> h_x(N);
  for (int i = 0; i < N; i++) {
    h_x[i] = static_cast<float>(rand() % 10 + 1);
  }

  // STEP 3: Create Output vector and pointers for GPU
  std::vector<float> h_y(M, 0.0f);

  // Allocate on GPU the correct size per each pointer
  int *d_csrRowPtr, *d_csrColInd;
  float *d_csrVal;
  CHECK_CUDA(cudaMalloc((void **)&d_csrRowPtr, (M + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_csrColInd, nnz * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_csrVal, nnz * sizeof(float)));

  // Copy data on GPU
  CHECK_CUDA(cudaMemcpy(d_csrRowPtr, h_csrRowPtr.data(), (M + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_csrColInd, h_csrColInd.data(), nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_csrVal, h_csrVal.data(), nnz * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Allocate vectors X and Y on the GPU ( Reminder Y=AX) A is a matrix, X is a
  // vector
  float *d_x, *d_y;
  CHECK_CUDA(cudaMalloc((void **)&d_x, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_y, M * sizeof(float)));
  // Copy Data
  CHECK_CUDA(
      cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // From: https://docs.nvidia.com/cuda/cusparse/#using-the-cusparse-api
  // The handle to the cuSPARSE library context is initialized using the
  // function and is explicitly passed to every subsequent library function
  // call. This allows the user to have more control over the library setup when
  // using multiple host threads and multiple GPUs.
  cusparseHandle_t handle;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  // CuSparse works with matrix and vector descriptors. This function
  // initializes the matrix descriptor in CSR
  cusparseSpMatDescr_t matA;
  CHECK_CUSPARSE(cusparseCreateCsr(
      &matA, M, N, nnz, d_csrRowPtr, d_csrColInd, d_csrVal, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  // CuSparse works with matrix and vector descriptors. This function
  // initializes the vector descriptors for Vec and Output
  cusparseDnVecDescr_t vecX, vecY;
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_32F));
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, M, d_y, CUDA_R_32F));

  // CuSparase allows to define some parameters
  // The Operation is Y = alpha * A(mxn) X(nx1) + beta(Y)
  // alpha = constant multiplied to every product - 1 in our case
  // beta = scalar multiplied to the pre-existent output value
  // TODO: There are multiple other parameters, needs to be checked as they are
  // not used in the guide for SpMV but may be useful
  float alpha = 1.0f;
  float beta = 0.0f;
  size_t bufferSize = 0;
  void *dBuffer = NULL;

  // Y = alpha * A * x
  // The function cusparseSpMV_bufferSize() returns the size of the
  // workspace needed by cusparseSpMV_preprocess() and cusparseSpMV()
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
      CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  // workspace allocation
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  const int iterations = 20;
  float totalTime = 0.0f;
  for (int i = 0; i <= iterations; i++) {
    std::cout << "Running Iteration: " << i << " over: " << iterations
              << std::endl;
    CHECK_CUDA(cudaMemset(d_y, 0, M * sizeof(float)));
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsedTime;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
    if (i > 0) totalTime += elapsedTime;
  }
  float averageTime = totalTime / iterations;
  std::cout << "Average time: " << averageTime << " ms" << std::endl;
  std::string matrix_name =
      matrixfile.substr(matrixfile.find_last_of("/\\") + 1);
  std::ofstream csv_file("time_sparse_matrix.csv", std::ios::app);
  if (!csv_file.is_open()) {
    std::cerr << "Error opening CSV file!" << std::endl;
  } else {
    csv_file << matrix_name << "," << averageTime << std::endl;
    csv_file.close();
  }
  CHECK_CUDA(
      cudaMemcpy(h_y.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUSPARSE(cusparseDestroySpMat(matA));
  CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
  CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
  CHECK_CUSPARSE(cusparseDestroy(handle));
  cudaFree(d_csrRowPtr);
  cudaFree(d_csrColInd);
  cudaFree(d_csrVal);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(dBuffer);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
