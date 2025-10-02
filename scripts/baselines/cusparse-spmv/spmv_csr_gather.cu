// To compile, use nvcc -O3 spmv_csr_gather.cu -o spmv_csr_gather
//
// To run
//
// 1. [Download a CSR Sparse Matrix] wget
// https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/italy_osm.tar.gz
// 2. [Untar] tar zxvf italy_osm.tar.gz
// 3. Run kernel: ./spmv-csr-gather italy_osm/italy_osm.mtx

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
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

/**
 * @brief Reads a CSR Sparse Matrix from .mtx format, see Sparse Suite Matrices.
 *
 * @param [in] mtxpath Input Sparse Suite matrix (.mtx file extension)
 * @param [out] row_ptr CSR row pointers
 * @param [out] col_idx CSR column indices.
 * @param [out] vals Sparse matrix non-zero values
 * @return int Returns number of columns or -1 if cannot open input .mtx file.
 */
int read_mtx_no_zeros(const std::string mtxpath, std::vector<int> &row_ptr,
                      std::vector<int> &col_idx, std::vector<float> &vals) {
  std::ifstream fin(mtxpath);
  if (!fin.is_open()) {
    return -1;
  }

  while (fin.peek() == '%') fin.ignore(2048, '\n');

  int M, N, L;
  fin >> M >> N >> L;

  std::vector<std::vector<int>> temp_cols(M);
  std::vector<std::vector<float>> temp_vals(M);

  for (int l = 0; l < L; ++l) {
    int m, n;
    float real_val = 1.0f;

    std::string line;
    std::getline(fin >> std::ws, line);
    std::istringstream iss(line);

    if (!(iss >> m >> n)) {
      continue;
    }

    if (!(iss >> real_val)) {
      real_val = 1;
    }

    if (real_val != 0.0f) {
      int row = m - 1;
      int col = n - 1;
      temp_cols[row].push_back(col);
      temp_vals[row].push_back(real_val);
    }
  }

  fin.close();

  row_ptr.clear();
  col_idx.clear();
  vals.clear();

  row_ptr.push_back(0);
  for (int i = 0; i < M; ++i) {
    col_idx.insert(col_idx.end(), temp_cols[i].begin(), temp_cols[i].end());
    vals.insert(vals.end(), temp_vals[i].begin(), temp_vals[i].end());
    row_ptr.push_back(static_cast<int>(col_idx.size()));
  }

  return N;
}

__global__ void spmv_csr_gather(int nnz, int *ptr, int *indices, float *data,
                                float *x, float *out_product) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < nnz) {
    out_product[idx] = data[idx] * x[indices[idx]];
  }
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
  const int N =
      read_mtx_no_zeros(matrixfile, h_csrRowPtr, h_csrColInd, h_csrVal);

  srand(time(NULL));
  const int M = h_csrRowPtr.size() - 1;
  const int nnz = h_csrVal.size();
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
  std::vector<float> h_out(nnz, 0.0f);

  // Allocate on GPU the correct size per each pointer
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

  float *d_x;
  CHECK_CUDA(cudaMalloc((void **)&d_x, N * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  float *d_out;
  CHECK_CUDA(cudaMalloc((void **)&d_out, nnz * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_out, 0, nnz * sizeof(float)));

  size_t bufferSize = 0;
  void *dBuffer = NULL;

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  const int iterations = 20;
  float totalTime = 0.0f;
  for (int i = 0; i <= iterations; i++) {
    std::cout << "Running Iteration: " << i << " over: " << iterations
              << std::endl;
    CHECK_CUDA(cudaMemset(d_out, 0, nnz * sizeof(float)));
    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    int block_size = (nnz + 1024 - 1) / 1024;
    spmv_csr_gather<<<block_size, 1024>>>(nnz, d_csrRowPtr, d_csrColInd,
                                          d_csrVal, d_x, d_out);
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsedTime;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
    if (i > 0) totalTime += elapsedTime;
  }

  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, nnz * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Make sure results are correct
  for (int i = 0; i < nnz; i++) {
    float expected = h_csrVal[i] * h_x[h_csrColInd[i]];
    assert(std::abs(h_out[i] - expected) < 1e-5);
  }

  float averageTime = totalTime / iterations;
  std::cout << "Average time: " << averageTime << " ms" << std::endl;
  std::string matrix_name =
      matrixfile.substr(matrixfile.find_last_of("/\\") + 1);
  std::ofstream csv_file("time_csr_gather_spmv.csv", std::ios::app);
  if (!csv_file.is_open()) {
    std::cerr << "Error opening CSV file!" << std::endl;
  } else {
    csv_file << matrix_name << "," << averageTime << "," << nnz << std::endl;
    csv_file.close();
  }

  cudaFree(d_csrRowPtr);
  cudaFree(d_csrColInd);
  cudaFree(d_csrVal);
  cudaFree(d_x);
  cudaFree(d_out);
  cudaFree(dBuffer);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
