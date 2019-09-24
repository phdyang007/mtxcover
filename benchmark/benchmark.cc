#include <cstring>

#include "common.h"
#include "data_reader.h"
#include "target_fn.h"

std::vector<std::string> test_datasets = {
    "../matrix/s1", 
};

std::vector<std::string> validation_sets = {
    "../dlresults/s1.txt",
};

/*
std::vector<std::string> test_datasets = {
    "../matrix/s1", "../matrix/s2", "../matrix/s3",  "../matrix/s4",
    "../matrix/s5", "../matrix/c1", "../matrix/c2",  "../matrix/c3",
    "../matrix/c4", "../matrix/c5", "../matrix/c6",  "../matrix/c7",
    "../matrix/c8", "../matrix/c9", "../matrix/c10",
};

std::vector<std::string> validation_sets = {
    "../dlresults/s1.txt", "../dlresults/s2.txt", "../dlresults/s3.txt",
    "../dlresults/s4.txt", "../dlresults/s5.txt", "../dlresults/c1.txt",
    "../dlresults/c2.txt", "../dlresults/c3.txt", "../dlresults/c4.txt",
    "../dlresults/c5.txt", "../dlresults/c6.txt", "../dlresults/c7.txt",
    "../dlresults/c8.txt", "../dlresults/c9.txt", "../dlresults/c10.txt",
};
*/

int main(int argc, char *argv[]) {
  bool validate = true;
  if (argc >= 2) {
    validate = strcmp(argv[1], "1") == 0;
  }
  int n = test_datasets.size();
  for (int i = 0; i < n; ++i) {
    const auto &tdataset = test_datasets[i];
    const auto &vset = validation_sets[i];
    std::vector<DataSet> dataset = ReadDataSetFromMatrixFolder(tdataset, vset);

    std::cout << "\n========================\n";

    // CPU
    std::cout << "\n>>> DataSet: " << tdataset
              << "    Matrix Count: " << dataset.size() << std::endl;
    std::cout << "-----------------------\nCPU BENCHMARK\n\n";
    {
      double core_ns = 0;
      for (auto &ds : dataset) {
        auto timer = Invoke(ImplVersion::ORIGINAL_CPU, false, &ds);
        core_ns += timer.GetCoreUsedNs();
        if (validate) {
          ValidateArray(ds.expected_result, ds.final_result);
        }
      }
      std::cout << "> Core Used NS: " << std::to_string(core_ns) << std::endl;
    }

    // GPU
    std::cout << "-----------------------\nGPU BENCHMARK\n\n";
    {
      double core_ns = 0;
      std::vector<DataSet> dataset = ReadDataSetFromMatrixFolder(tdataset, vset);
      for (auto &ds : dataset) {
        auto timer = Invoke(ImplVersion::ORIGINAL_GPU, true, &ds);
        core_ns += timer.GetCoreUsedNs();
        // std::cout << "> Load to GPU Used NS: "
        //           << std::to_string(timer.GetDataLoadingNs()) << std::endl;
        if (validate) {
          ValidateArray(ds.expected_result, ds.final_result);
        }
      }
      std::cout << "> Core Used NS: " << std::to_string(core_ns) << std::endl;
    }

    // GPU_MG
    {
      std::cout << "-----------------------\nGPU MG BENCHMARK\n\n";
      DataSets datasets = CombineDatasets(dataset);
      auto timer = Invoke(ImplVersion::ORIGINAL_GPU_MG, false, &datasets);

      std::cout << "> Core Used NS: " << std::to_string(timer.GetCoreUsedNs())
                << std::endl;
      std::cout << "> Load to GPU Used NS: "
                << std::to_string(timer.GetDataLoadingNs()) << std::endl;

      if (validate) {
        ValidateArray(datasets.expected_result, datasets.final_result);
      }
    }

    std::cout << "========================\n\n\n";
  }
  return 0;
}
