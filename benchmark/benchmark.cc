#include "common.h"
#include "data_reader.h"
#include "target_fn.h"
#include <cstring>
#include <fstream>
#include <thread>

/*
std::vector<std::string> test_datasets = {
    "../matrix/c1",
};

std::vector<std::string> validation_sets = {
    "../dlresults/c1.txt",
};

*/
///*
///*
std::vector<std::string> test_datasets = {
    "../matrix/s1", "../matrix/s2", "../matrix/s3",  "../matrix/s4",
    "../matrix/s5", "../matrix/c1", "../matrix/c2",  "../matrix/c3",
    "../matrix/c4", "../matrix/c5", "../matrix/c6",  "../matrix/c7",
    "../matrix/c8", "../matrix/c9", "../matrix/c10",
};
//*/
/*
std::vector<std::string> test_datasets = {
    "../matrix/s3",
};
*/
std::vector<std::string> validation_sets = {
    "../dlresults/s1.txt", "../dlresults/s2.txt", "../dlresults/s3.txt",
    "../dlresults/s4.txt", "../dlresults/s5.txt", "../dlresults/c1.txt",
    "../dlresults/c2.txt", "../dlresults/c3.txt", "../dlresults/c4.txt",
    "../dlresults/c5.txt", "../dlresults/c6.txt", "../dlresults/c7.txt",
    "../dlresults/c8.txt", "../dlresults/c9.txt", "../dlresults/c10.txt",
};

std::vector<std::string> cpu_results = {
    "../cpuresults/s1.txt", "../cpuresults/s2.txt", "../cpuresults/s3.txt",
    "../cpuresults/s4.txt", "../cpuresults/s5.txt", "../cpuresults/c1.txt",
    "../cpuresults/c2.txt", "../cpuresults/c3.txt", "../cpuresults/c4.txt",
    "../cpuresults/c5.txt", "../cpuresults/c6.txt", "../cpuresults/c7.txt",
    "../cpuresults/c8.txt", "../cpuresults/c9.txt", "../cpuresults/c10.txt",
};

std::vector<std::string> gpu_results = {
    "../gpuresults/s1.txt", "../gpuresults/s2.txt", "../gpuresults/s3.txt",
    "../gpuresults/s4.txt", "../gpuresults/s5.txt", "../gpuresults/c1.txt",
    "../gpuresults/c2.txt", "../gpuresults/c3.txt", "../gpuresults/c4.txt",
    "../gpuresults/c5.txt", "../gpuresults/c6.txt", "../gpuresults/c7.txt",
    "../gpuresults/c8.txt", "../gpuresults/c9.txt", "../gpuresults/c10.txt",
};

//*/

int main(int argc, char *argv[]) {
  bool validate = false;
  bool dumpout = true;
  if (argc >= 2) {
    validate = strcmp(argv[1], "1") == 0;
  }
  int n = test_datasets.size();
  int debug_file = 1;
  int debug_graph = 1691;
  for (int i = 0; i < n; ++i) {
    // if (i != debug_file - 1) {
    //   continue;
    // }
    const auto &tdataset = test_datasets[i];
    const auto &vset = cpu_results[i];
    std::vector<DataSet> dataset = ReadDataSetFromMatrixFolder(tdataset, vset);

// std::cout << "\n========================\n";

// CPU
// std::cout << "\n>>> DataSet: " << tdataset
//           << "    Matrix Count: " << dataset.size() << std::endl;
#ifdef CPU
    // std::cout << "----------------------s-\nCPU BENCHMARK\n\n";
    double time_bgn = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          clock_type::now().time_since_epoch())
                          .count();
    {
      // double core_ns = 0;
      int j = 0;
//#pragma omp parallel for num_threads(16) schedule(dynamic, 8)
      for (unsigned int idx = 0; idx < dataset.size(); ++idx) {
        auto &ds = dataset[idx];
        // j++;
        // if (j != debug_graph) {
        //   continue;
        // }
        // std::cout<<"dataset is "<<cpu_results[i]<<" component id is
        // "<<j<<std::endl;
        auto timer = Invoke(ImplVersion::ORIGINAL_CPU, false, &ds);
        // core_ns += timer.GetCoreUsedNs();
        if (validate) {
          ValidateArray(ds.expected_result, ds.final_result);
        }
        if (dumpout) {
          std::fstream of(cpu_results[i], std::ios::out | std::ios::app);
          if (of.is_open()) {
            for (int i = 0; i < ds.final_result.size(); i++) {
              of << ds.final_result[i] << ' ';
            }
          }
          of << std::endl;
          of.close();
        }
      }
      double time_end = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            clock_type::now().time_since_epoch())
                            .count();
      double core_ns = time_end - time_bgn;
      std::cout << std::to_string(core_ns) << std::endl;
    }
#endif
    // GPU
#ifdef GPU
//     std::cout << "-----------------------\nGPU BENCHMARK\n\n";
//     {
//       double core_ns = 0;
// #pragma omp parallel for num_threads(16) schedule(dynamic, 8)
//       for (unsigned int idx = 0; idx < dataset.size(); ++idx) {
//         // j++;
//         // if(j!=debug_graph){continue;}
//         // std::cout<<"dataset is "<<cpu_results[i]<<" component id is
//         // "<<j<<std::endl;
//         auto timer = Invoke(ImplVersion::ORIGINAL_GPU, false, &ds);
//         core_ns += timer.GetCoreUsedNs();
//         // std::cout << "> Load to GPU Used NS: "
//         //           << std::to_string(timer.GetDataLoadingNs()) <<
//         // std::endl;
//         if (validate) {
//           ValidateArray(ds.expected_result, ds.final_result);
//         }
//       }
//       std::cout << "> Core Used NS: " << std::to_string(core_ns)
//                 << "   s:" << std::to_string(core_ns * 10e-10) << std::endl;
//     }
#endif
    // GPU_MG
#ifdef GPUMG
    {
      // std::cout << "-----------------------\nGPU MG BENCHMARK\n\n";
      DataSets datasets = CombineDatasets(dataset);
      auto timer = Invoke(ImplVersion::ORIGINAL_GPU_MG, false, &datasets);

      // std::cout << "> Core Used NS: " <<
      // std::to_string(timer.GetCoreUsedNs())
      //           << "   s:" << std::to_string(timer.GetCoreUsedNs() * 10e-10)
      //           << std::endl;
      // std::cout << "> Load to GPU Used NS: "
      //           << std::to_string(timer.GetDataLoadingNs()) << std::endl;
      std::cout << std::to_string(timer.GetCoreUsedNs()) << std::endl;

      if (validate) {
        ValidateArray(datasets.expected_result, datasets.final_result);
      }
    }
// std::cout << "========================\n\n\n";
#endif
  }

  return 0;
}
