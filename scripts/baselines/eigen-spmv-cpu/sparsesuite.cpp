/**
 * @file sparsesuite.cpp
 * @brief SpMV using Eigen's SparseMatrix<T, RowMajor> container.
 * @date 2025-03-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <Eigen/Sparse>
#include <chrono>
#include <fast_matrix_market/app/Eigen.hpp>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace Eigen;

using SparseMatrixRowMajor = SparseMatrix<float, RowMajor>;

/**
 * @brief Read a Sparse Suite Matrix and run SpMv on a random vector.
 *
 * @param argc Number of input arguments.
 * @param argv List of input arguments.
 * @return Status code
 */
int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <matrixpath>" << std::endl;
    return 1;
  }
  int n_execs = 10;
  std::string matrixfile = argv[1];
  std::cout << "Handling matrix: " << matrixfile << std::endl;

  std::cout << "Reading matrix: " << matrixfile << std::endl;

  std::ifstream mtx_file(matrixfile);

  SparseMatrixRowMajor mat;
  fast_matrix_market::read_matrix_market_eigen(mtx_file, mat);

  std::cout << "Matrix loaded successfully!" << std::endl;
  std::cout << "Matrix size: " << mat.rows() << " x " << mat.cols()
            << std::endl;

  VectorXf x(mat.cols());
  x.setRandom();

  const std::string delimiter = "/";
  const std::string token = matrixfile.substr(matrixfile.rfind(delimiter) + 1);
  const std::string newDelimiter = ".";
  const std::string matname = token.substr(0, token.find(newDelimiter));

  std::ofstream fout("bench_results_eigen_spmv_" + matname + ".csv",
                     std::ios::app);
  fout << "benchname,size,time_us" << std::endl;

  for (int i = 0; i < n_execs; i++) {
    const auto start = std::chrono::high_resolution_clock::now();
    VectorXf y = mat * x;
    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    fout << matname << "," << mat.nonZeros() << "," << elapsed_us.count()
         << std::endl;
  }
  fout.close();
  return 0;
}
