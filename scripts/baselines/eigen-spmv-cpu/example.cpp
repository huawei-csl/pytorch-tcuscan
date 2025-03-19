#include <Eigen/Sparse>
#include <iostream>

int main() {
  // Create a sparse matrix (using the Eigen Sparse Matrix format)
  Eigen::SparseMatrix<double> mat(4, 4);

  // Manually filling the sparse matrix
  // (In a real scenario, you might load this from a file or generate it
  // dynamically)
  mat.insert(0, 0) = 5.0;
  mat.insert(1, 1) = -3.0;
  mat.insert(2, 2) = 9.0;
  mat.insert(3, 3) = -4.0;
  mat.insert(0, 1) = 6.0;

  // Create a vector to multiply with the matrix
  Eigen::VectorXd x(4);
  x << 1.0, 2.0, 3.0, 4.0;

  // Perform the multiplication: result = mat * x
  Eigen::VectorXd result = mat * x;

  // Print the result vector
  std::cout << "Resulting vector: \n" << result << std::endl;

  return 0;
}
