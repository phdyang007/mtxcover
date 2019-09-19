
#include "data_reader.h"

#include <fstream>

std::vector<DataSet> ReadDataSet(const std::string &cofig_file) {
  std::ifstream file(cofig_file);
  std::vector<DataSet> datasets;
  int n;
  file >> n;
  std::string matrix_file, col_file;
  for (int i = 0; i < n; ++i) {
    DataSet dataset;
    file >> dataset.vertex_num;
    file >> dataset.total_dl_matrix_row_num;
    file >> dataset.total_dl_matrix_col_num;

    dataset.dl_matrix.resize(dataset.total_dl_matrix_row_num *
                             dataset.total_dl_matrix_col_num);
    dataset.col_group.resize(dataset.total_dl_matrix_col_num, 0);

    file >> matrix_file;
    file >> col_file;
    {
      //   std::cout << "matrix_file: " << matrix_file << std::endl;
      std::ifstream m_file(matrix_file);
      for (int i = 0; i < dataset.total_dl_matrix_row_num; ++i) {
        for (int j = 0; j < dataset.total_dl_matrix_col_num; ++j) {
          m_file >> dataset.dl_matrix[i * dataset.total_dl_matrix_col_num + j];
        }
      }
    }
    {
      //   std::cout << "col_file: " << col_file << std::endl;
      std::ifstream m_file(col_file);
      for (int i = 0; i < dataset.total_dl_matrix_col_num; ++i) {
        m_file >> dataset.col_group[i];
      }
    }
    datasets.push_back(dataset);
  }
  return datasets;
}

DataSets CombineDatasets(const std::vector<DataSet> &dataset) {
  DataSets datasets;
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
    datasets.dl_matrix.insert(datasets.dl_matrix.end(),
                              dataset[i].dl_matrix.begin(),
                              dataset[i].dl_matrix.end());
    datasets.col_group.insert(datasets.col_group.end(),
                              dataset[i].col_group.begin(),
                              dataset[i].col_group.end());

    offset_matrix +=
        dataset[i].total_dl_matrix_col_num * dataset[i].total_dl_matrix_row_num;
    offset_row += dataset[i].total_dl_matrix_row_num;
    offset_col += dataset[i].total_dl_matrix_col_num;
  }
  return datasets;
}
