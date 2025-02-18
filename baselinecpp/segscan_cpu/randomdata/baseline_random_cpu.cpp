#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

void segmented_scan(const std::vector<float> &data,
                    const std::vector<int> &flags, std::vector<float> &output) {
  const size_t n = data.size();
  output[0] = data[0];
  for (size_t i = 1; i < n; i++) {
    if (flags[i] == 1) {
      output[i] = data[i];  // Start new segment
    } else {
      output[i] = output[i - 1] + data[i];  // Continue segment
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <size_of_array> <density>"
              << std::endl;
    return 1;
  }

  constexpr int REPEATS = 1;
  int N = std::atoi(argv[1]);
  double density = std::atof(argv[2]);
  if (N <= 0 || density < 0.0 || density > 1.0) {
    std::cerr << "Error: Size of array must be a positive integer and density "
                 "must be between 0 and 1.";
    return 1;
  }

  srand(time(NULL));

  std::vector<float> data(N);
  std::vector<int> flags(N);
  std::vector<float> output(N);

  // Initialize data and flags randomly
  std::cout << "Generating Input";
  for (int i = 0; i < N; i++) {
    data[i] = rand() % 10 + 1;  // Random values between 1 and 10
  }
  std::cout << "\n";

  std::cout << "Generating Flags:";
  for (int i = 0; i < N; i++) {
    flags[i] = ((rand() / (double)RAND_MAX) < density)
                   ? 1
                   : 0;  // Set segment start based on density
  }
  std::cout << "\n";

  std::cout << "Data Size: " << data.size() << std::endl;
  std::cout << "Flag Size: " << flags.size() << std::endl;
  std::cout << "Output Size: " << output.size() << std::endl;

  // Measure execution time
  const auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < REPEATS; i++) {
    segmented_scan(data, flags, output);
  }
  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "Output Scan:  ";
  long sum = 0;
  for (int i = 0; i < N; i++) {
    sum += output[i];
  }
  std::cout << "Ouput sum: " << sum << std::endl;

  std::cout << "Elapsed Time: " << elapsed_us.count() << " us.\n";

  return 0;
}
