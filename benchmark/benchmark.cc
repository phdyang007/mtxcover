#include "common.h"
#include "data_reader.h"
#include "target_fn.h"

std::vector<std::string> test_configs = {"configs/config.txt"};

int main() {
  // GPU
  std::cout << "========================\nGPU BENCHMARK\n\n";
  for (const auto &config_file : test_configs) {
    std::vector<DataSet> dataset = ReadDataSet(config_file);
    for (auto &ds : dataset) {
      auto timer = Invoke(ImplVersion::ORIGINAL_GPU, false, &ds);

      std::cout << "> Core Used NS: " << std::to_string(timer.GetCoreUsedNs())
                << std::endl;
      std::cout << "> Load to GPU Used NS: "
                << std::to_string(timer.GetDataLoadingNs()) << std::endl;
    }
  }

  std::cout << "========================\nGPU MG BENCHMARK\n\n";
  // GPU_MG
  for (const auto &config_file : test_configs) {
    std::vector<DataSet> dataset = ReadDataSet(config_file);
    DataSets datasets = CombineDatasets(dataset);
    auto timer = Invoke(ImplVersion::ORIGINAL_GPU_MG, false, &datasets);

    std::cout << "> Core Used NS: " << std::to_string(timer.GetCoreUsedNs())
              << std::endl;
    std::cout << "> Load to GPU Used NS: "
              << std::to_string(timer.GetDataLoadingNs()) << std::endl;
  }

  return 0;
}
