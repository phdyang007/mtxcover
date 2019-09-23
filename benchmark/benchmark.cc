#include "common.h"
#include "data_reader.h"
#include "target_fn.h"

std::vector<std::string> test_datasets = {
    // "../matrix/simple",
    "../matrix/s1", "../matrix/s2", "../matrix/s3",  "../matrix/s4",
    "../matrix/s5", "../matrix/c1", "../matrix/c2",  "../matrix/c3",
    "../matrix/c4", "../matrix/c5", "../matrix/c6",  "../matrix/c7",
    "../matrix/c8", "../matrix/c9", "../matrix/c10",
};

int main() {
  for (const auto &tdataset : test_datasets) {
    std::cout << "\n========================\n";
    std::cout << "\n>>> DataSet: " << tdataset << std::endl;

    // CPU
    std::cout << "-----------------------\nCPU BENCHMARK\n\n";
    {
      std::vector<DataSet> dataset = ReadDataSetFromMatrixFolder(tdataset);
      for (auto &ds : dataset) {
        auto timer = Invoke(ImplVersion::ORIGINAL_CPU, false, &ds);

        std::cout << "> Core Used NS: " << std::to_string(timer.GetCoreUsedNs())
                  << std::endl;
        std::cout << "> Load to GPU Used NS: "
                  << std::to_string(timer.GetDataLoadingNs()) << std::endl;
      }
    }
    // // GPU
    // std::cout << "-----------------------\nGPU BENCHMARK\n\n";
    // {
    //   std::vector<DataSet> dataset = ReadDataSetFromMatrixFolder(tdataset);
    //   for (auto &ds : dataset) {
    //     auto timer = Invoke(ImplVersion::ORIGINAL_GPU, false, &ds);

    //     std::cout << "> Core Used NS: " <<
    //     std::to_string(timer.GetCoreUsedNs())
    //               << std::endl;
    //     std::cout << "> Load to GPU Used NS: "
    //               << std::to_string(timer.GetDataLoadingNs()) << std::endl;
    //   }
    // }

    std::cout << "\n>>> DataSet: " << tdataset << std::endl;
    // GPU_MG

    {
      std::cout << "-----------------------\nGPU MG BENCHMARK\n\n";
      std::vector<DataSet> dataset = ReadDataSetFromMatrixFolder(tdataset);
      DataSets datasets = CombineDatasets(dataset);
      auto timer = Invoke(ImplVersion::ORIGINAL_GPU_MG, false, &datasets);

      std::cout << "> Core Used NS: " << std::to_string(timer.GetCoreUsedNs())
                << std::endl;
      std::cout << "> Load to GPU Used NS: "
                << std::to_string(timer.GetDataLoadingNs()) << std::endl;
    }

    std::cout << "========================\n\n\n";
  }
  return 0;
}
