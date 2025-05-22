/**
 * @file kml_spmv_sparsesuite.cpp
 * @brief SpMV using Kunpeng's KML sparseblas SpMV.
 * @date 2025-05-21
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <kspblas.h>

#include <chrono>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
using namespace std;

template <typename IT, typename VT>
struct TripletMatrix {
  int64_t nrows = 0, ncols = 0;
  std::vector<IT> rows, cols;
  std::vector<VT> vals;  // or int64_t, float, std::complex<double>, etc.
};

template <typename IT, typename VT>
struct CSRMatrix {
  int64_t nrows = 0, ncols = 0;
  std::vector<IT> indptr;
  std::vector<IT> indices;
  std::vector<VT> vals;
};

struct PairHash {
  std::size_t operator()(const std::pair<int, int> &p) const {
    return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
  }
};

int triplet_to_csr(const TripletMatrix<int, float> &smat,
                   std::vector<int> &row_ptr, std::vector<int> &col_idx,
                   std::vector<float> &vals) {
  int M = smat.nrows;
  int NNZ = smat.vals.size();

  std::unordered_map<std::pair<int, int>, float, PairHash> entry_map;

  for (int i = 0; i < NNZ; ++i) {
    int r = smat.rows[i];
    int c = smat.cols[i];
    float v = smat.vals[i];

    if (r < 0 || r >= M) {
      std::cerr << "[ERROR] Out of Bound index row: " << r << "\n";
      return 1;
    }

    if (v != 0.0f) {
      entry_map[{r, c}] = v;
    }
  }

  row_ptr.assign(M + 1, 0);
  for (const auto &[key, _] : entry_map) {
    int r = key.first;
    row_ptr[r + 1]++;
  }

  for (int i = 1; i <= M; ++i) {
    row_ptr[i] += row_ptr[i - 1];
  }

  int unique_nnz = entry_map.size();
  col_idx.resize(unique_nnz);
  vals.resize(unique_nnz);

  std::vector<int> offset = row_ptr;
  for (const auto &[key, value] : entry_map) {
    int r = key.first;
    int c = key.second;
    int dest = offset[r]++;
    col_idx[dest] = c;
    vals[dest] = value;
  }

  return 0;
}

void convert_to_kml_format(const std::vector<int> &row_ptr,
                           const std::vector<int> &col_idx,
                           std::vector<int> &indx, std::vector<int> &pntrb,
                           std::vector<int> &pntre) {
  int m = row_ptr.size() - 1;

  indx.clear();
  pntrb.clear();
  pntre.clear();

  for (int i = 0; i < m; ++i) {
    int start = row_ptr[i];
    int end = row_ptr[i + 1];

    pntrb.push_back(static_cast<int>(indx.size()) + 1);  // 1-based
    for (int j = start; j < end; ++j) {
      indx.push_back(col_idx[j] + 1);  // convert to 1-based indexing
    }
    pntre.push_back(static_cast<int>(indx.size()) + 1);  // exclusive, 1-based
  }
}

void dummy_kml_spmv_example() {
  kml_sparse_operation_t opt = KML_SPARSE_OPERATION_NON_TRANSPOSE;

  KML_INT m = 4;
  KML_INT k = 4;
  float alpha = 0.5;
  float beta = 1.2;
  string matdescra = "G00F";  // General matrix with one-based indexing
  float val[9] = {2, -3, 7, 1, -6, 8, -4, 5, 9};
  KML_INT indx[9] = {1, 2, 4, 3, 4, 1, 3, 4, 1};
  KML_INT pntrb[4] = {1, 4, 6, 9};
  KML_INT pntre[4] = {4, 6, 9, 10};
  float x[4] = {1, 3, -2, 5};
  float y[4] = {-1, 1, 5, 3};

  // SpMV operation: y = A * x
  kml_sparse_status_t status = kml_sparse_scsrmv(
      opt, m, k, alpha, matdescra.c_str(), val, indx, pntrb, pntre, x, beta, y);

  cout << " KML_SPARSE_STATUS: " << status << endl;

  printf("Result y = A * x:\n");
  for (int i = 0; i < m; ++i) {
    printf("y[%d] = %.2f\n", i, y[i]);
  }
}

std::vector<float> generate_random_vector(int size, float min = -1.0f,
                                          float max = 1.0f) {
  std::random_device rd;
  std::mt19937 gen(rd());  // Mersenne Twister RNG
  std::uniform_real_distribution<float> dist(min, max);

  std::vector<float> x(size);
  for (int i = 0; i < size; ++i) {
    x[i] = dist(gen);
  }
  return x;
}

int64_t run_spmv_kml(std::vector<int> &row_ptr, std::vector<int> &col_idx,
                     std::vector<float> &vals) {
  kml_sparse_operation_t opt = KML_SPARSE_OPERATION_NON_TRANSPOSE;
  KML_INT m = row_ptr.size() - 1;
  KML_INT k = m;
  float alpha = 1.0f;
  float beta = 0.0f;
  std::string matdescra =
      "G00F";  // General matrix, one-based indexing (KML expects 1-based)

  std::vector<float> x = generate_random_vector(k);
  std::vector<float> y(m, 0.0f);
  std::vector<int> indx, pntrb, pntre;
  convert_to_kml_format(row_ptr, col_idx, indx, pntrb, pntre);

  auto start = std::chrono::high_resolution_clock::now();

  kml_sparse_status_t status = kml_sparse_scsrmv(
      opt, m, k, alpha, matdescra.c_str(), vals.data(), indx.data(),
      pntrb.data(), pntre.data(), x.data(), beta, y.data());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();

  if (status == 0) {
    std::cout << " KML_SPARSE_STATUS: SUCCESS" << endl;

  } else {
    std::cout << " KML_SPARSE_STATUS: ERROR" << endl;
  }

  return duration_us;
}

/**
 * @brief Read a Sparse Suite Matrix and run SpMv on a random vector using
 * Kunpeng's KML C sparse BLAS.
 *
 * @param argc Number of input arguments.
 * @param argv List of input arguments.
 * @return Status code
 */
int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <matrixpath>" << endl;
    return 1;
  }

  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<float> values;
  std::string matrixfile = static_cast<std::string>(argv[1]);
  std::string matrix_name =
      matrixfile.substr(matrixfile.find_last_of("/\\") + 1);
  size_t dot_pos = matrix_name.find_last_of('.');
  if (dot_pos != std::string::npos && matrix_name.substr(dot_pos) == ".mtx") {
    matrix_name = matrix_name.substr(0, dot_pos);
  }
  std::cout << "Reading matrix: " << matrix_name << endl;

  ifstream mtx_istream(matrixfile);
  TripletMatrix<int, float> smat;
  fast_matrix_market::read_matrix_market_triplet(
      mtx_istream, smat.nrows, smat.ncols, smat.rows, smat.cols, smat.vals);
  std::cout << "Matrix loaded successfully!" << std::endl;
  triplet_to_csr(smat, row_ptr, col_idx, values);
  std::cout << "Triple to CSR converted successfully!" << std::endl;
  int64_t exec_time = run_spmv_kml(row_ptr, col_idx, values);

  std::string filename = "bench_results_boostkit_spmv.csv";
  bool file_is_empty = !std::filesystem::exists(filename) ||
                       std::filesystem::file_size(filename) == 0;

  std::ofstream fout(filename, std::ios::app);
  if (!fout) {
    std::cerr << "[ERROR] Cannot Open file: " << filename << "\n";
    return -1;
  }
  if (file_is_empty) {
    fout << "benchname,size,time_us" << std::endl;
  }

  fout << matrix_name << "," << values.size() << "," << exec_time << std::endl;

  fout.close();
  return 0;
}
