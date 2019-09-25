#pragma once

#include <cassert>

#include <chrono>
#include <iostream>
#include <vector>

using clock_type = std::chrono::high_resolution_clock;

struct DataSet {
  int vertex_num;
  int total_dl_matrix_row_num;
  int total_dl_matrix_col_num;
  std::vector<int> dl_matrix;
  std::vector<int> col_group;
  std::vector<int> next_row;
  std::vector<int> next_col;
  std::vector<int> expected_result;
  std::vector<int> final_result;
};

struct DataSets {
  int graph_count;
  std::vector<int> vertex_num;
  std::vector<int> total_dl_matrix_row_num;
  std::vector<int> total_dl_matrix_col_num;
  std::vector<int> offset_matrix;
  std::vector<int> offset_row;
  std::vector<int> offset_col;
  std::vector<int> dl_matrix;
  std::vector<int> col_group;

  std::vector<int> next_row;
  std::vector<int> next_col;

  std::vector<int> expected_result;
  std::vector<int> final_result;
};

class MeasureTimer {
public:
  inline void StartSetupTime() { setup_start_ns_ = Now(); }
  inline void EndSetupTime() { setup_ns_ += Now() - setup_start_ns_; }

  inline void StartCoreTime() { core_start_ns_ = Now(); }
  inline void EndCoreTime() { core_used_ns_ += Now() - core_start_ns_; }

  inline void StartDataLoadTime() { data_loading_start_ns_ = Now(); }
  inline void EndDataLoadTime() {
    data_loading_ns_ += Now() - data_loading_start_ns_;
  }

  inline double GetSetupNs() const { return setup_ns_; }
  inline double GetCoreUsedNs() const { return core_used_ns_; }
  inline double GetDataLoadingNs() const { return data_loading_ns_; }

private:
  double setup_start_ns_ = 0;
  double core_start_ns_ = 0;
  double data_loading_start_ns_ = 0;

  double setup_ns_ = 0;
  double core_used_ns_ = 0;
  double data_loading_ns_ = 0;

  inline double Now() const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock_type::now().time_since_epoch())
        .count();
  }
};

template <typename T>
void ValidateArray(const std::vector<T> &a, const std::vector<T> &b) {
  int an = a.size();
  int bn = b.size();
  // std::cout << "expected n: " << an << "  final n: " << bn << std::endl;
  assert(an == bn);
  // for (int i = 0; i < an; ++i) {
  //   std::cout << a[i] << " <> " << b[i] << ", ";
  // }
  // std::cout << std::endl;

  for (int i = 0; i < an; ++i) {
    assert(a[i] == b[i]);
  }
}
