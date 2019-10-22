#pragma once

#include <cuda_profiler_api.h>
#include "common.h"
#include "../cpu/MatrixCover.h"
#include "../gpu-mg/MatrixCoverGPU.cuh"
#include "../gpu/MatrixCoverGPU.cuh"

enum class ImplVersion { ORIGINAL_CPU, ORIGINAL_GPU, ORIGINAL_GPU_MG };

#define THRESHOLD 500
// struct DataSets {
//     int graph_count;
//     std::vector<int> total_dl_matrix_row_num[2]={276,276};
//     std::vector<int> total_dl_matrix_col_num[2]={29,29};
//     std::vector<int> offset_matrix[2]={0, 276*29};
//     std::vector<int> offset_row[2]={0,276};
//     std::vector<int> offset_col[2]={0,29};
//     std::vector<int> dl_matrix;
// vertex_num
// };
template <typename MCSolverTraitsType>
MeasureTimer Invoke_ORIGINAL_CPU(DataSet<MCSolverTraitsType> *dataset, bool print_result) {
  MeasureTimer timer;

  int total_col = dataset->total_dl_matrix_col_num;
  int total_row = dataset->total_dl_matrix_row_num;

  std::vector<typename MCSolverTraitsType::ResultType> results(total_row, 0);
  std::vector<int> deleted_cols(total_col, 0);
  typename MCSolverTraitsType::MatrixType **dl_matrix = new typename MCSolverTraitsType::MatrixType *[total_row];
  for (int i = 0; i < total_row; i++) {
    dl_matrix[i] = new typename MCSolverTraitsType::MatrixType [total_col];
    for (int j = 0; j < total_col; j++) {
      dl_matrix[i][j] = dataset->dl_matrix[i * total_col + j];
    }
  }
  int hard_conflict_threshold = THRESHOLD;
  timer.StartCoreTime();
  mc_solver<MCSolverTraitsType>(dl_matrix, results.data(), deleted_cols.data(),
            dataset->col_group.data(), dataset->vertex_num, total_row,
            total_col, hard_conflict_threshold);
  timer.EndCoreTime();

  dataset->final_result.clear();
  for (int i = 0; i < total_row; i++) {
    if (results[i] > 0) {
      if (i + 1 > 3 * dataset->vertex_num) {
        dataset->final_result.push_back(i + 2);
      } else {
        dataset->final_result.push_back(i + 1);
      }
    }
  }

  if (print_result) {
    int conflict_count = 0;
    for (int i = 0; i < total_row; i++) {
      std::cout << results[i] << ' ';
    }
    std::cout << std::endl;
    for (int i = 0; i < total_row; i++) {
      if (results[i] > 0) {
        std::cout << i << ' ';
      }
    }
    std::cout << std::endl;
    for (int i = 0; i < total_col; i++) {
      if (deleted_cols[i] == -1) {
        conflict_count++;
      }
    }

    std::cout << "Conflict Num is " << conflict_count / 3 << std::endl;
  }

  for (int i = 0; i < total_row; i++) {
    delete[] dl_matrix[i];
  }
  delete[] dl_matrix;

  return timer;
}

template <typename MCSolverTraitsType>
MeasureTimer Invoke_ORIGINAL_GPU(DataSet<MCSolverTraitsType> *dataset, bool print_result) {
  MeasureTimer timer;

  timer.StartDataLoadTime();

  int total_col = dataset->total_dl_matrix_col_num;
  int total_row = dataset->total_dl_matrix_row_num;

  std::vector<int> deleted_cols(total_col, 0);
  std::vector<int> deleted_rows(total_row, 0);

  int thread_size = 32;

  int conflict_count = 0;
  int vertex_num = dataset->vertex_num;
  int vertex_num_gpu = vertex_num;
  int total_dl_matrix_row_num = total_row;
  int total_dl_matrix_col_num = total_col;
  int total_dl_matrix_row_num_gpu = total_row;
  int total_dl_matrix_col_num_gpu = total_col;

  // allocate necessary vectors and matrix on GPU
  typename MCSolverTraitsType::MatrixType *dl_matrix_gpu = nullptr;
  int *deleted_cols_gpu = nullptr;
  typename MCSolverTraitsType::ColGroupType *col_group_gpu = nullptr;
  typename MCSolverTraitsType::ResultType *results_gpu = nullptr;
  typename MCSolverTraitsType::ConflictCountType *conflict_count_gpu = nullptr;
  int *deleted_rows_gpu = nullptr;
  typename MCSolverTraitsType::RowGroupType *row_group_gpu = nullptr;

  // dl_matrix_gpu = new int *[total_dl_matrix_row_num];
  cudaMalloc(&dl_matrix_gpu,
             sizeof(typename MCSolverTraitsType::MatrixType) * total_dl_matrix_row_num * total_dl_matrix_col_num);
  cudaMemcpy(dl_matrix_gpu, dataset->dl_matrix.data(),
             sizeof(typename MCSolverTraitsType::MatrixType) * total_dl_matrix_row_num * total_dl_matrix_col_num,
             cudaMemcpyHostToDevice);

  cudaMalloc(&deleted_cols_gpu, sizeof(int) * total_dl_matrix_col_num);
  cudaMalloc(&col_group_gpu, sizeof(typename MCSolverTraitsType::ColGroupType) * total_dl_matrix_col_num);
  cudaMalloc(&results_gpu, sizeof(typename MCSolverTraitsType::ResultType) * total_dl_matrix_row_num);
  cudaMalloc(&conflict_count_gpu, sizeof(typename MCSolverTraitsType::ConflictCountType) * total_dl_matrix_col_num);
  cudaMalloc(&deleted_rows_gpu, sizeof(int) * total_dl_matrix_row_num);
  cudaMalloc(&row_group_gpu, sizeof(typename MCSolverTraitsType::RowGroupType) * total_dl_matrix_row_num);

  std::vector<typename MCSolverTraitsType::RowGroupType> row_group(total_dl_matrix_row_num_gpu, 0);
  // get col and row group
  gpu::init_vectors<<<1, thread_size>>>(row_group_gpu,
                                        total_dl_matrix_row_num_gpu);
  cudaMemcpy(row_group.data(), row_group_gpu,
             sizeof(typename MCSolverTraitsType::RowGroupType) * total_dl_matrix_row_num_gpu, cudaMemcpyDeviceToHost);
  if (print_result) {
    std::cout << "print row group" << std::endl;
    for (int i = 0; i < total_dl_matrix_row_num; i++) {
      std::cout << row_group[i] << ' ';
    }
    std::cout << std::endl;
  }

  gpu::get_vertex_row_group<MCSolverTraitsType><<<1, thread_size>>>(
      row_group_gpu, dl_matrix_gpu, vertex_num_gpu, total_dl_matrix_row_num_gpu,
      total_dl_matrix_col_num_gpu);
  cudaMemcpy(row_group.data(), row_group_gpu,
             sizeof(typename MCSolverTraitsType::RowGroupType) * total_dl_matrix_row_num_gpu, cudaMemcpyDeviceToHost);
  if (print_result) {
    std::cout << "print row group" << std::endl;
    for (int i = 0; i < total_dl_matrix_row_num; i++) {
      std::cout << row_group[i] << ' ';
    }
    std::cout << std::endl;
  }

  cudaMemcpy(col_group_gpu, dataset->col_group.data(),
             sizeof(typename MCSolverTraitsType::ColGroupType) * total_dl_matrix_col_num, cudaMemcpyHostToDevice);

  timer.EndDataLoadTime();

  timer.StartCoreTime();
  cudaProfilerStart();
  gpu::mc_solver<MCSolverTraitsType>(dl_matrix_gpu, results_gpu, deleted_cols_gpu, deleted_rows_gpu,
                 col_group_gpu, row_group_gpu, conflict_count_gpu,
                 vertex_num_gpu, total_dl_matrix_row_num_gpu,
                 total_dl_matrix_col_num_gpu);
  cudaDeviceSynchronize();
  cudaProfilerStop();

  timer.EndCoreTime();

  std::vector<typename MCSolverTraitsType::ResultType> results(total_dl_matrix_row_num);
  cudaMemcpy(results.data(), results_gpu, sizeof(typename MCSolverTraitsType::ResultType) * total_dl_matrix_row_num,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(deleted_cols.data(), deleted_cols_gpu,
             sizeof(int) * total_dl_matrix_col_num, cudaMemcpyDeviceToHost);
  dataset->final_result.clear();
  for (int i = 0; i < total_row; i++) {
    if (results[i] > 0) {
      // std::cout<<"debug"<<dataset->final_result.empty()<<std::endl;
      if (i + 1 > 3 * dataset->vertex_num) {
        dataset->final_result.push_back(i + 2);
      } else {
        dataset->final_result.push_back(i + 1);
      }
    }
  }
  if (print_result) {
    for (int i = 0; i < total_dl_matrix_row_num; i++) {
      std::cout << results[i] << ' ';
    }
    std::cout << std::endl;
    for (int i = 0; i < total_dl_matrix_row_num; i++) {
      if (results[i] > 0) {
        std::cout << i << ' ';
      }
    }
    std::cout << std::endl;
    for (int i = 0; i < total_dl_matrix_col_num; i++) {
      if (deleted_cols[i] == -1) {
        conflict_count++;
      }
    }

    std::cout << "Conflict Num is " << conflict_count / 3 << std::endl;
  }

  cudaFree(dl_matrix_gpu);
  cudaFree(deleted_cols_gpu);
  cudaFree(col_group_gpu);
  cudaFree(results_gpu);
  cudaFree(conflict_count_gpu);
  cudaFree(deleted_rows_gpu);
  cudaFree(row_group_gpu);

  return timer;
}

template <typename MCSolverTraitsType>
MeasureTimer Invoke_ORIGINAL_GPU_MG(DataSets<MCSolverTraitsType> *datasets, bool print_result) {
  MeasureTimer timer;

  int total_row = 0, total_col = 0;
  int n = datasets->graph_count;
  for (int i = 0; i < n; ++i) {
    total_row += datasets->total_dl_matrix_row_num[i];
    total_col += datasets->total_dl_matrix_col_num[i];
  }
  int total_matrix = datasets->dl_matrix.size();

  std::vector<int> deleted_cols(total_col, 0);
  std::vector<int> deleted_rows(total_row, 0);
  std::vector<typename MCSolverTraitsType::ConflictCountType> conflict_count(n, 0);

  timer.StartDataLoadTime();
  typename MCSolverTraitsType::MatrixType *dl_matrix_gpu;
  typename MCSolverTraitsType::NextColType *next_col_gpu;
  typename MCSolverTraitsType::NextRowType *next_row_gpu;
  typename MCSolverTraitsType::ResultType *results_gpu;
  int *conflict_edge_gpu = nullptr;
  cudaMalloc(&dl_matrix_gpu, sizeof(typename MCSolverTraitsType::MatrixType) * total_matrix);
  //cudaMalloc(&conflict_edge_gpu, sizeof(int) * 2 * n);
  cudaMalloc(&next_col_gpu, sizeof(typename MCSolverTraitsType::NextColType) * total_matrix);
  cudaMalloc(&next_row_gpu, sizeof(typename MCSolverTraitsType::NextRowType) * total_matrix);
  cudaMemcpy(dl_matrix_gpu, datasets->dl_matrix.data(),
             sizeof(typename MCSolverTraitsType::MatrixType) * total_matrix, cudaMemcpyHostToDevice);
  cudaMemcpy(next_col_gpu, datasets->next_col.data(),
             sizeof(typename MCSolverTraitsType::NextColType) * total_matrix, cudaMemcpyHostToDevice);
  cudaMemcpy(next_row_gpu, datasets->next_row.data(),
             sizeof(typename MCSolverTraitsType::NextRowType) * total_matrix, cudaMemcpyHostToDevice);
  cudaMalloc(&results_gpu, sizeof(typename MCSolverTraitsType::ResultType) * total_row);

  int *deleted_cols_gpu = nullptr;
  int *deleted_rows_gpu = nullptr;
  typename MCSolverTraitsType::ColGroupType *col_group_gpu = nullptr;
  typename MCSolverTraitsType::RowGroupType *row_group_gpu = nullptr;
  typename MCSolverTraitsType::ConflictCountType *conflict_count_gpu = nullptr;

  //cudaMalloc(&deleted_cols_gpu, sizeof(int) * total_col);
  //cudaMalloc(&deleted_rows_gpu, sizeof(int) * total_row);
  cudaMalloc(&col_group_gpu, sizeof(typename MCSolverTraitsType::ColGroupType) * total_col);
  cudaMalloc(&row_group_gpu, sizeof(typename MCSolverTraitsType::RowGroupType) * total_row);
  //cudaMalloc(&conflict_count_gpu, sizeof(int) * total_col);
  cudaMemcpy(col_group_gpu, datasets->col_group.data(), sizeof(typename MCSolverTraitsType::ColGroupType) * total_col,
             cudaMemcpyHostToDevice);

  int *vertex_num_gpu;
  int *total_dl_matrix_col_num_gpu;
  int *total_dl_matrix_row_num_gpu;

  cudaMalloc(&vertex_num_gpu, sizeof(int) * n);
  cudaMalloc(&total_dl_matrix_col_num_gpu, sizeof(int) * n);
  cudaMalloc(&total_dl_matrix_row_num_gpu, sizeof(int) * n);

  cudaMemcpy(vertex_num_gpu, datasets->vertex_num.data(), sizeof(int) * n,
             cudaMemcpyHostToDevice);
  cudaMemcpy(total_dl_matrix_col_num_gpu,
             datasets->total_dl_matrix_col_num.data(), sizeof(int) * n,
             cudaMemcpyHostToDevice);
  cudaMemcpy(total_dl_matrix_row_num_gpu,
             datasets->total_dl_matrix_row_num.data(), sizeof(int) * n,
             cudaMemcpyHostToDevice);

  int *offset_col_gpu = nullptr;
  int *offset_row_gpu = nullptr;
  int *offset_matrix_gpu = nullptr;
  int *max_gpu = nullptr;
  cudaMalloc(&offset_col_gpu, sizeof(int) * n);
  cudaMalloc(&offset_row_gpu, sizeof(int) * n);
  cudaMalloc(&offset_matrix_gpu, sizeof(int) * n);
  //cudaMalloc(&max_gpu, sizeof(int) * n);

  cudaMemcpy(offset_col_gpu, datasets->offset_col.data(), sizeof(int) * n,
             cudaMemcpyHostToDevice);
  cudaMemcpy(offset_row_gpu, datasets->offset_row.data(), sizeof(int) * n,
             cudaMemcpyHostToDevice);
  cudaMemcpy(offset_matrix_gpu, datasets->offset_matrix.data(), sizeof(int) * n,
             cudaMemcpyHostToDevice);

  int *search_depth_gpu = nullptr;
  int *selected_row_id_gpu = nullptr;
  int *current_conflict_count_gpu = nullptr;
  int *conflict_node_id_gpu = nullptr;
  int *conflict_col_id_gpu = nullptr;
  int *existance_of_candidate_rows_gpu = nullptr;

  //cudaMalloc(&search_depth_gpu, sizeof(int) * n);
  //cudaMalloc(&selected_row_id_gpu, sizeof(int) * n);
  //cudaMalloc(&current_conflict_count_gpu, sizeof(int) * n);
  //cudaMalloc(&conflict_node_id_gpu, sizeof(int) * n);
  //cudaMalloc(&conflict_col_id_gpu, sizeof(int) * n);
  //cudaMalloc(&existance_of_candidate_rows_gpu, sizeof(int) * n);

  timer.EndDataLoadTime();

  int hard_conflict_threshold = THRESHOLD;
  int graph_per_block=1;
  int thread_count = 32;
  dim3 thread_size(thread_count,graph_per_block);
  cudaDeviceSynchronize();

  timer.StartCoreTime();

  gpu_mg::init_vertex_group<MCSolverTraitsType><<<n,32>>>(
    row_group_gpu, dl_matrix_gpu, vertex_num_gpu, 
    total_dl_matrix_col_num_gpu, total_dl_matrix_row_num_gpu, 
    offset_row_gpu, offset_matrix_gpu,n);

  cudaProfilerStart();
  gpu_mg::mc_solver<MCSolverTraitsType><<<n/graph_per_block+1, thread_size>>>(
      dl_matrix_gpu, next_col_gpu, next_row_gpu, results_gpu, deleted_cols_gpu,
      deleted_rows_gpu, col_group_gpu, row_group_gpu, conflict_count_gpu,
      vertex_num_gpu, total_dl_matrix_row_num_gpu, total_dl_matrix_col_num_gpu,
      offset_col_gpu, offset_row_gpu, offset_matrix_gpu, search_depth_gpu,
      selected_row_id_gpu, current_conflict_count_gpu, conflict_node_id_gpu,
      conflict_col_id_gpu, existance_of_candidate_rows_gpu, conflict_edge_gpu,
      max_gpu, n, hard_conflict_threshold, graph_per_block);
  cudaDeviceSynchronize();
  cudaProfilerStop();

  timer.EndCoreTime();

  std::vector<typename MCSolverTraitsType::ResultType> results(total_row, 0);
  cudaMemcpy(results.data(), results_gpu, sizeof(typename MCSolverTraitsType::ResultType) * total_row,
             cudaMemcpyDeviceToHost);
  datasets->final_result.clear();
  for (int k = 0; k < n; k++) {
    for (int i = 0; i < datasets->total_dl_matrix_row_num[k]; i++) {
      if (results[datasets->offset_row[k] + i] > 0) {
        if (i + 1 > 3 * datasets->vertex_num[k]) {
          datasets->final_result.push_back(i + 2);
        } else {
          datasets->final_result.push_back(i + 1);
        }
      }
    }
  }
  /*
  if (print_result) {
    cudaMemcpy(deleted_cols.data(), deleted_cols_gpu, sizeof(int) * total_col,
               cudaMemcpyDeviceToHost);
    for (int k = 0; k < n; k++) {
      for (int i = 0; i < datasets->total_dl_matrix_row_num[k]; i++) {
        std::cout << results[datasets->offset_row[k] + i] << ' ';
      }
      std::cout << std::endl;
      for (int i = 0; i < datasets->total_dl_matrix_row_num[k]; i++) {
        if (results[datasets->offset_row[k] + i] > 0) {
          std::cout << i << ' ';
        }
      }
      std::cout << std::endl;
      for (int i = 0; i < datasets->total_dl_matrix_col_num[k]; i++) {
        if (deleted_cols[datasets->offset_col[k] + i] == -1) {
          conflict_count[k]++;
        }
      }

      // 3 is the number of color
      std::cout << "Conflict Num is " << conflict_count[k] / 3 << std::endl;
    }
  }
  */

  cudaFree(dl_matrix_gpu);
  cudaFree(next_col_gpu);
  cudaFree(next_row_gpu);
  cudaFree(results_gpu);
  //cudaFree(deleted_cols_gpu);
  //cudaFree(deleted_rows_gpu);
  cudaFree(col_group_gpu);
  cudaFree(row_group_gpu);
  //cudaFree(conflict_count_gpu);
  //cudaFree(max_gpu);
  cudaFree(vertex_num_gpu);
  cudaFree(total_dl_matrix_col_num_gpu);
  cudaFree(total_dl_matrix_row_num_gpu);
  cudaFree(offset_col_gpu);
  cudaFree(offset_row_gpu);
  cudaFree(offset_matrix_gpu);
  //cudaFree(search_depth_gpu);
  //cudaFree(selected_row_id_gpu);
  //cudaFree(current_conflict_count_gpu);
  //cudaFree(conflict_col_id_gpu);
  //cudaFree(conflict_node_id_gpu);
  //cudaFree(existance_of_candidate_rows_gpu);
  //cudaFree(conflict_edge_gpu);
  return timer;
}

template <typename MCSolverTraitsType>
MeasureTimer Invoke(const ImplVersion version, bool print_result,
                    DataSet<MCSolverTraitsType> *dataset) {
  MeasureTimer default_timer;
  switch (version) {
  case ImplVersion::ORIGINAL_CPU:
    return Invoke_ORIGINAL_CPU(dataset, print_result);
  case ImplVersion::ORIGINAL_GPU:
    return Invoke_ORIGINAL_GPU(dataset, print_result);
  default:
    std::cout << "Not Impl yet" << std::endl;
    return default_timer;
  }
}

template <typename MCSolverTraitsType>
MeasureTimer Invoke(const ImplVersion version, bool print_result,
                    DataSets<MCSolverTraitsType> *datasets) {
  MeasureTimer default_timer;
  switch (version) {
  case ImplVersion::ORIGINAL_GPU_MG:
    return Invoke_ORIGINAL_GPU_MG(datasets, print_result);
  default:
    std::cout << "Not Impl yet" << std::endl;
    return default_timer;
  }
}
