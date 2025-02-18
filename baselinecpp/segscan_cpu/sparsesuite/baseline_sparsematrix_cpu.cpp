#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
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

int read_mtx(std::string mtxpath, std::vector<float> *x, std::vector<int> *f) {
  std::ifstream fin(mtxpath);
  int M, N, L;
  while (fin.peek() == '%') fin.ignore(2048, '\n');

  fin >> M >> N >> L;
  x->resize(L);
  f->resize(L, 0);  // Inizializza f con 0, lungo M
  int old = -1;
  for (int l = 0; l < L; l++) {
    int m, n;
    float data;
    fin >> m >> n >> data;
    (*x)[l] = static_cast<float>(data);
    if (m != old) {
      (*f)[l] = 1;
      old = m;
    } else {
      (*f)[l] = 0;
    }
  }
  fin.close();
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Passed arguments: " << argc << std::endl;
    std::cerr << "Usage: " << argv[0] << " <matrixpath>" << std::endl;
    return 1;
  }

  constexpr int REPEATS = 50;
  std::vector<float> data;
  std::vector<int> flags;
  std::vector<float> output;
  std::string matrixfile = static_cast<std::string>(argv[1]);
  read_mtx(matrixfile, &data, &flags);
  output.resize(data.size());
  std::cout << "Data Size: " << data.size() << std::endl;
  std::cout << "Flag Size: " << flags.size() << std::endl;
  std::cout << "Output Size: " << output.size() << std::endl;
  std::cout << "Number of Repeats: " << REPEATS << std::endl;
  // Measure execution time
  auto tot_time = 0;
  for (int i = 0; i < REPEATS; i++) {
    const auto start = std::chrono::high_resolution_clock::now();
    segmented_scan(data, flags, output);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    tot_time += elapsed_us.count();
  }

  std::cout << "Compute Output Scan ";
  long sum = 0;

  for (int i = 0; i < data.size(); i++) sum += output[i];

  std::cout << "Ouput sum: " << sum << std::endl;
  const auto final_time = tot_time / REPEATS;
  std::cout << "Elapsed Time: " << final_time << " us.\n";
  const std::string delimiter = "/";
  const std::string token = matrixfile.substr(matrixfile.rfind(delimiter) + 1);
  std::cout << "Token: " << token << std::endl;
  const std::string newDelimiter = ".";
  const std::string token2 = token.substr(0, token.find(newDelimiter));
  std::ofstream fout("bench_results_baseline_cpu_seg_scan_sc_" + token2 +
                     ".csv");
  fout << "benchname,operator,dtype,size,time_us" << std::endl;
  fout << token2 << ",baseline_cpu_segscan,fp32," << data.size() << ","
       << final_time << std::endl;
  return 0;
}
