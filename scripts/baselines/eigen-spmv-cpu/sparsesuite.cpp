#include <Eigen/Sparse>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace Eigen;

using SparseMatrixRowMajor = SparseMatrix<float, RowMajor>;

int read_mtx(const std::string& mtxpath, SparseMatrixRowMajor& mat) {
  std::ifstream fin(mtxpath);
  if (!fin.is_open()) {
    std::cerr << "Error opening file: " << mtxpath << std::endl;
    return -1;
  }

  std::string line;
  while (std::getline(fin, line)) {
    if (line[0] == '%')
      continue;
    else {
      std::stringstream ss(line);
      int M, N, L;
      ss >> M >> N >> L;

      mat.resize(M, N);

      for (int l = 0; l < L; ++l) {
        int row, col;
        float value;
        fin >> row >> col >> value;
        // Adjust for zero-indexing
        mat.insert(row - 1, col - 1) = value;
      }
      break;
    }
  }

  fin.close();
  return 0;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <matrixpath>" << std::endl;
    return 1;
  }
  int n_execs = 10;
  std::string matrixfile = argv[1];

  SparseMatrixRowMajor mat;

  if (read_mtx(matrixfile, mat) != 0) {
    std::cerr << "Error reading matrix from file!" << std::endl;
    return 1;
  }

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
