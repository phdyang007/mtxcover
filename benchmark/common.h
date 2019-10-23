#pragma once

#include <cassert>

#include <chrono>
#include <iostream>
#include <vector>

using clock_type = std::chrono::high_resolution_clock;

/// @param dl_matrix binary matrix, can be unsigned char, even boolean 
/// @param next_col can be unsigned char 
/// @param next_row can be unsigned short 
/// @param results can be unsigned short 
/// @param col_group can be char 
/// @param row_group can be unsigned char 
/// @param conflict_count must be unsigned int 
/// @param vertex_num a number indicating how many vertices 
/// @param total_dl_matrix_row_num a number, M
/// @param total_dl_matrix_col_num a number, N
/// @param offset_col CSR 
/// @param offset_row CSR 
/// @param offset_matrix CSR
/// @param graph_count total number of graphs 
/// @param graph_per_block graphs that can be solved on one threadblock
template <int MaxNumRows, int MaxNumCols>
struct MCSolverTraits
{
    typedef int MatrixType;
    typedef int NextColType; 
    typedef int NextRowType; 
    typedef int ResultType; 
    typedef int ColGroupType; 
    typedef int RowGroupType; 
    typedef int ConflictCountType; 

    static constexpr int max_num_rows = MaxNumRows; 
    static constexpr int max_num_cols = MaxNumCols;
};

template <>
struct MCSolverTraits<512, 128>
{
    typedef unsigned char MatrixType;
    typedef unsigned char NextColType; 
    typedef unsigned short NextRowType; 
    typedef unsigned short ResultType; 
    typedef char ColGroupType; 
    typedef unsigned char RowGroupType; 
    typedef int ConflictCountType; 

    static constexpr int max_num_rows = 512; 
    static constexpr int max_num_cols = 128;
};


template <typename MCSolverTraitsType>
struct DataSet {
  int vertex_num;
  int total_dl_matrix_row_num;
  int total_dl_matrix_col_num;
  std::vector<typename MCSolverTraitsType::MatrixType> dl_matrix;
  std::vector<typename MCSolverTraitsType::ColGroupType> col_group;
  std::vector<typename MCSolverTraitsType::NextRowType> next_row;
  std::vector<typename MCSolverTraitsType::NextColType> next_col;
  std::vector<typename MCSolverTraitsType::ResultType> expected_result;
  std::vector<typename MCSolverTraitsType::ResultType> final_result;
};

typedef MCSolverTraits<512, 512> InstantiateType1; 
typedef MCSolverTraits<512, 128> InstantiateType2; 

template <typename MCSolverTraitsType>
struct DataSets {
  int graph_count;
  std::vector<int> vertex_num;
  std::vector<int> total_dl_matrix_row_num;
  std::vector<int> total_dl_matrix_col_num;
  std::vector<int> offset_matrix;
  std::vector<int> offset_row;
  std::vector<int> offset_col;
  std::vector<typename MCSolverTraitsType::MatrixType> dl_matrix;
  std::vector<typename MCSolverTraitsType::ColGroupType> col_group;

  std::vector<typename MCSolverTraitsType::NextRowType> next_row;
  std::vector<typename MCSolverTraitsType::NextColType> next_col;

  std::vector<typename MCSolverTraitsType::ResultType> expected_result;
  std::vector<typename MCSolverTraitsType::ResultType> final_result;
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
  std::cout << "expected n: " << an << "  final n: " << bn << std::endl;
  assert(an == bn);
  for (int i = 0; i < an; ++i) {
    std::cout << a[i] << " <> " << b[i] << ", ";
    assert(a[i] == b[i]);
  }
  std::cout << std::endl;

  //for (int i = 0; i < an; ++i) {
  //  assert(a[i] == b[i]);
  //}
}
