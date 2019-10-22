#pragma once

#include <iostream>
#include <algorithm>
#include <fstream>

#include "common.h"

template <typename MCSolverTraitsType>
std::vector<DataSet<MCSolverTraitsType> >
ReadDataSetFromMatrixFolder(const std::string &matrix_folder,
                            const std::string &result_txt) {
  std::string n_data_txt = matrix_folder + "/n_data.txt";
  std::string col_txt = matrix_folder + "/col_cnt.txt";
  std::string row_txt = matrix_folder + "/row_cnt.txt";
  std::string col_group_txt = matrix_folder + "/col.txt";
  std::string vertex_txt = matrix_folder + "/vetex.txt";
  std::string matrix_txt = matrix_folder + "/matrix.txt";
  std::string validation_txt = result_txt;

  int n = 0;
  {
    std::ifstream m_file(n_data_txt);
    m_file >> n;
  }

  std::vector<DataSet<MCSolverTraitsType> > out;
  std::ifstream col_file(col_txt);
  std::ifstream col_group_file(col_group_txt);
  std::ifstream row_file(row_txt);
  std::ifstream vertex_file(vertex_txt);
  std::ifstream matrix_file(matrix_txt);
  std::ifstream validation_file(validation_txt);

  for (int k = 0; k < n; ++k) {
      DataSet<MCSolverTraitsType> dataset;
      int tmp = std::numeric_limits<int>::max(); 
      col_file >> tmp; 
      dataset.total_dl_matrix_col_num = tmp;
      row_file >> tmp; 
      dataset.total_dl_matrix_row_num = tmp;
      vertex_file >> tmp; 
      dataset.vertex_num = tmp;

      assert(dataset.total_dl_matrix_row_num < std::numeric_limits<typename MCSolverTraitsType::NextRowType>::max());
      assert(dataset.total_dl_matrix_col_num < std::numeric_limits<typename MCSolverTraitsType::NextColType>::max());
      int nm = dataset.total_dl_matrix_row_num * dataset.total_dl_matrix_col_num;
      dataset.dl_matrix.resize(nm);
      dataset.next_col.resize(nm);
      dataset.next_row.resize(nm);
      dataset.col_group.resize(dataset.total_dl_matrix_col_num, 0);
      dataset.expected_result.resize(dataset.vertex_num, 0);

      for (int i = 0; i < dataset.total_dl_matrix_col_num; ++i) {
          int tmp = std::numeric_limits<int>::max(); 
          col_group_file >> tmp; 
          assert(tmp <= (int)std::numeric_limits<typename MCSolverTraitsType::ColGroupType>::max());
          dataset.col_group[i] = tmp;
      }

      for (int i = 0; i < dataset.total_dl_matrix_row_num; ++i) {
          for (int j = 0; j < dataset.total_dl_matrix_col_num; ++j) {
              int tmp = std::numeric_limits<int>::max(); 
              matrix_file >> tmp; 
              assert(tmp <= (int)std::numeric_limits<typename MCSolverTraitsType::MatrixType>::max());
              dataset.dl_matrix[i * dataset.total_dl_matrix_col_num + j] = tmp;
          }
      }

      for (int i = 0; i < dataset.vertex_num; ++i) {
          int tmp = std::numeric_limits<int>::max(); 
          validation_file >> tmp; 
          assert(tmp <= (int)std::numeric_limits<typename MCSolverTraitsType::ResultType>::max());
          dataset.expected_result[i] = tmp;
      }
      std::sort(dataset.expected_result.begin(), dataset.expected_result.end());

      for (int i = 0; i < dataset.total_dl_matrix_row_num; ++i) {
          int last_col = dataset.total_dl_matrix_col_num;
          for (int j = dataset.total_dl_matrix_col_num - 1; j >= 0; --j) {
              dataset.next_col[i * dataset.total_dl_matrix_col_num + j] =
                  last_col - j;
              if (dataset.dl_matrix[i * dataset.total_dl_matrix_col_num + j] == 1) {
                  last_col = j;
              }
          }
      }
      for (int j = 0; j < dataset.total_dl_matrix_col_num; ++j) {
          int last_row = dataset.total_dl_matrix_row_num;
          for (int i = dataset.total_dl_matrix_row_num - 1; i >= 0; --i) {
              dataset.next_row[j * dataset.total_dl_matrix_row_num + i] =
                  last_row - i;
              if (dataset.dl_matrix[i * dataset.total_dl_matrix_col_num + j] == 1) {
                  last_row = i;
              }
          }
      }

      out.push_back(dataset);
  }
  return out;
}

template <typename MCSolverTraitsType>
DataSets<MCSolverTraitsType> CombineDatasets(const std::vector<DataSet<MCSolverTraitsType> > &dataset) {
  DataSets<MCSolverTraitsType> datasets;
  int n = dataset.size();
  datasets.graph_count = n;
  int offset_col = 0, offset_row = 0, offset_matrix = 0;
  for (int i = 0; i < n; ++i) {
    datasets.vertex_num.push_back(dataset[i].vertex_num);
    datasets.total_dl_matrix_row_num.push_back(
        dataset[i].total_dl_matrix_row_num);
    datasets.total_dl_matrix_col_num.push_back(
        dataset[i].total_dl_matrix_col_num);
    if (i == 0) {
      datasets.offset_matrix.push_back(0);
      datasets.offset_row.push_back(0);
      datasets.offset_col.push_back(0);
    } else {
      datasets.offset_matrix.push_back(offset_matrix);
      datasets.offset_row.push_back(offset_row);
      datasets.offset_col.push_back(offset_col);
    }
    datasets.next_col.insert(datasets.next_col.end(),
                             dataset[i].next_col.begin(),
                             dataset[i].next_col.end());
    datasets.next_row.insert(datasets.next_row.end(),
                             dataset[i].next_row.begin(),
                             dataset[i].next_row.end());
    datasets.dl_matrix.insert(datasets.dl_matrix.end(),
                              dataset[i].dl_matrix.begin(),
                              dataset[i].dl_matrix.end());
    datasets.col_group.insert(datasets.col_group.end(),
                              dataset[i].col_group.begin(),
                              dataset[i].col_group.end());

    datasets.expected_result.insert(datasets.expected_result.end(),
                                    dataset[i].expected_result.begin(),
                                    dataset[i].expected_result.end());

    offset_matrix +=
        dataset[i].total_dl_matrix_col_num * dataset[i].total_dl_matrix_row_num;
    offset_row += dataset[i].total_dl_matrix_row_num;
    offset_col += dataset[i].total_dl_matrix_col_num;
  }
  return datasets;
}
