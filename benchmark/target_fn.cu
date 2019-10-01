#include "target_fn.h"

#include <cuda_profiler_api.h>

#include "../cpu/MatrixCover.h"
#include "../gpu-mg/MatrixCoverGPU.cuh"
#include "../gpu/MatrixCoverGPU.cuh"

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
MeasureTimer Invoke_ORIGINAL_CPU(DataSet *dataset, bool print_result) {
  MeasureTimer timer;

  int total_col = dataset->total_dl_matrix_col_num;
  int total_row = dataset->total_dl_matrix_row_num;

  std::vector<int> results(total_row, 0);
  std::vector<int> deleted_cols(total_col, 0);
  int **dl_matrix = new int *[total_row];
  for (int i = 0; i < total_row; i++) {
    dl_matrix[i] = new int[total_col];
    for (int j = 0; j < total_col; j++) {
      dl_matrix[i][j] = dataset->dl_matrix[i * total_col + j];
    }
  }

  timer.StartCoreTime();
  mc_solver(dl_matrix, results.data(), deleted_cols.data(),
            dataset->col_group.data(), dataset->vertex_num, total_row,
            total_col);
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

MeasureTimer Invoke_ORIGINAL_GPU(DataSet *dataset, bool print_result) {
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
  int *dl_matrix_gpu;
  int *deleted_cols_gpu;
  int *col_group_gpu;
  int *results_gpu;
  int *conflict_count_gpu;
  int *deleted_rows_gpu;
  int *row_group_gpu;

  // dl_matrix_gpu = new int *[total_dl_matrix_row_num];
  cudaMalloc(&dl_matrix_gpu,
             sizeof(int) * total_dl_matrix_row_num * total_dl_matrix_col_num);
  cudaMemcpy(dl_matrix_gpu, dataset->dl_matrix.data(),
             sizeof(int) * total_dl_matrix_row_num * total_dl_matrix_col_num,
             cudaMemcpyHostToDevice);

  cudaMalloc(&deleted_cols_gpu, sizeof(int) * total_dl_matrix_col_num);
  cudaMalloc(&col_group_gpu, sizeof(int) * total_dl_matrix_col_num);
  cudaMalloc(&results_gpu, sizeof(int) * total_dl_matrix_row_num);
  cudaMalloc(&conflict_count_gpu, sizeof(int) * total_dl_matrix_col_num);
  cudaMalloc(&deleted_rows_gpu, sizeof(int) * total_dl_matrix_row_num);
  cudaMalloc(&row_group_gpu, sizeof(int) * total_dl_matrix_row_num);

  std::vector<int> row_group(total_dl_matrix_row_num_gpu, 0);
  // get col and row group
  gpu::init_vectors<<<1, thread_size>>>(row_group_gpu,
                                        total_dl_matrix_row_num_gpu);
  cudaMemcpy(row_group.data(), row_group_gpu,
             sizeof(int) * total_dl_matrix_row_num_gpu, cudaMemcpyDeviceToHost);
  if (print_result) {
    std::cout << "print row group" << std::endl;
    for (int i = 0; i < total_dl_matrix_row_num; i++) {
      std::cout << row_group[i] << ' ';
    }
    std::cout << std::endl;
  }

  gpu::get_vertex_row_group<<<1, thread_size>>>(
      row_group_gpu, dl_matrix_gpu, vertex_num_gpu, total_dl_matrix_row_num_gpu,
      total_dl_matrix_col_num_gpu);
  cudaMemcpy(row_group.data(), row_group_gpu,
             sizeof(int) * total_dl_matrix_row_num_gpu, cudaMemcpyDeviceToHost);
  if (print_result) {
    std::cout << "print row group" << std::endl;
    for (int i = 0; i < total_dl_matrix_row_num; i++) {
      std::cout << row_group[i] << ' ';
    }
    std::cout << std::endl;
  }

  cudaMemcpy(col_group_gpu, dataset->col_group.data(),
             sizeof(int) * total_dl_matrix_col_num, cudaMemcpyHostToDevice);

  timer.EndDataLoadTime();

  timer.StartCoreTime();
  cudaProfilerStart();
  gpu::mc_solver(dl_matrix_gpu, results_gpu, deleted_cols_gpu, deleted_rows_gpu,
                 col_group_gpu, row_group_gpu, conflict_count_gpu,
                 vertex_num_gpu, total_dl_matrix_row_num_gpu,
                 total_dl_matrix_col_num_gpu);
  cudaDeviceSynchronize();
  cudaProfilerStop();

  timer.EndCoreTime();

  std::vector<int> results(total_dl_matrix_row_num);
  cudaMemcpy(results.data(), results_gpu, sizeof(int) * total_dl_matrix_row_num,
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

MeasureTimer Invoke_ORIGINAL_GPU_MG(DataSets *datasets, bool print_result) {
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
  std::vector<int> conflict_count(n, 0);

  timer.StartDataLoadTime();
  int *dl_matrix_gpu;
  int *next_col_gpu;
  int *next_row_gpu;
  int *results_gpu;
  int *conflict_edge_gpu;
  cudaMalloc(&dl_matrix_gpu, sizeof(int) * total_matrix);
  cudaMalloc(&conflict_edge_gpu, sizeof(int) * 2 * n);
  cudaMalloc(&next_col_gpu, sizeof(int) * total_matrix);
  cudaMalloc(&next_row_gpu, sizeof(int) * total_matrix);
  cudaMemcpy(dl_matrix_gpu, datasets->dl_matrix.data(),
             sizeof(int) * total_matrix, cudaMemcpyHostToDevice);
  cudaMemcpy(next_col_gpu, datasets->next_col.data(),
             sizeof(int) * total_matrix, cudaMemcpyHostToDevice);
  cudaMemcpy(next_row_gpu, datasets->next_row.data(),
             sizeof(int) * total_matrix, cudaMemcpyHostToDevice);
  cudaMalloc(&results_gpu, sizeof(int) * total_row);

  int *deleted_cols_gpu;
  int *deleted_rows_gpu;
  int *col_group_gpu;
  int *row_group_gpu;
  int *conflict_count_gpu;

  cudaMalloc(&deleted_cols_gpu, sizeof(int) * total_col);
  cudaMalloc(&deleted_rows_gpu, sizeof(int) * total_row);
  cudaMalloc(&col_group_gpu, sizeof(int) * total_col);
  cudaMalloc(&row_group_gpu, sizeof(int) * total_row);
  cudaMalloc(&conflict_count_gpu, sizeof(int) * total_col);
  cudaMemcpy(col_group_gpu, datasets->col_group.data(), sizeof(int) * total_col,
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

  int *offset_col_gpu;
  int *offset_row_gpu;
  int *offset_matrix_gpu;
  int *max_gpu;
  cudaMalloc(&offset_col_gpu, sizeof(int) * n);
  cudaMalloc(&offset_row_gpu, sizeof(int) * n);
  cudaMalloc(&offset_matrix_gpu, sizeof(int) * n);
  cudaMalloc(&max_gpu, sizeof(int) * n);

  cudaMemcpy(offset_col_gpu, datasets->offset_col.data(), sizeof(int) * n,
             cudaMemcpyHostToDevice);
  cudaMemcpy(offset_row_gpu, datasets->offset_row.data(), sizeof(int) * n,
             cudaMemcpyHostToDevice);
  cudaMemcpy(offset_matrix_gpu, datasets->offset_matrix.data(), sizeof(int) * n,
             cudaMemcpyHostToDevice);

  int *search_depth_gpu;
  int *selected_row_id_gpu;
  int *current_conflict_count_gpu;
  int *conflict_node_id_gpu;
  int *conflict_col_id_gpu;
  int *existance_of_candidate_rows_gpu;

  cudaMalloc(&search_depth_gpu, sizeof(int) * n);
  cudaMalloc(&selected_row_id_gpu, sizeof(int) * n);
  cudaMalloc(&current_conflict_count_gpu, sizeof(int) * n);
  cudaMalloc(&conflict_node_id_gpu, sizeof(int) * n);
  cudaMalloc(&conflict_col_id_gpu, sizeof(int) * n);
  cudaMalloc(&existance_of_candidate_rows_gpu, sizeof(int) * n);

  timer.EndDataLoadTime();

  int hard_conflict_threshold = 500;
  int thread_size = 32;
  cudaDeviceSynchronize();
  gpu_mg::init_vertex_group<<<n,thread_size>>>(
    row_group_gpu, dl_matrix_gpu, vertex_num_gpu, 
    total_dl_matrix_col_num_gpu, total_dl_matrix_row_num_gpu, 
    offset_row_gpu, offset_matrix_gpu,n);

  cudaDeviceSynchronize();
  timer.StartCoreTime();

  cudaProfilerStart();
  gpu_mg::mc_solver<<<n, thread_size>>>(
      dl_matrix_gpu, next_col_gpu, next_row_gpu, results_gpu, deleted_cols_gpu,
      deleted_rows_gpu, col_group_gpu, row_group_gpu, conflict_count_gpu,
      vertex_num_gpu, total_dl_matrix_row_num_gpu, total_dl_matrix_col_num_gpu,
      offset_col_gpu, offset_row_gpu, offset_matrix_gpu, search_depth_gpu,
      selected_row_id_gpu, current_conflict_count_gpu, conflict_node_id_gpu,
      conflict_col_id_gpu, existance_of_candidate_rows_gpu, conflict_edge_gpu,
      max_gpu, n, hard_conflict_threshold);
  cudaDeviceSynchronize();
  cudaProfilerStop();

  timer.EndCoreTime();

  std::vector<int> results(total_row, 0);
  cudaMemcpy(results.data(), results_gpu, sizeof(int) * total_row,
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

  cudaFree(dl_matrix_gpu);
  cudaFree(next_col_gpu);
  cudaFree(next_row_gpu);
  cudaFree(results_gpu);
  cudaFree(deleted_cols_gpu);
  cudaFree(deleted_rows_gpu);
  cudaFree(col_group_gpu);
  cudaFree(row_group_gpu);
  cudaFree(conflict_count_gpu);
  cudaFree(max_gpu);
  cudaFree(vertex_num_gpu);
  cudaFree(total_dl_matrix_col_num_gpu);
  cudaFree(total_dl_matrix_row_num_gpu);
  cudaFree(offset_col_gpu);
  cudaFree(offset_row_gpu);
  cudaFree(offset_matrix_gpu);
  cudaFree(search_depth_gpu);
  cudaFree(selected_row_id_gpu);
  cudaFree(current_conflict_count_gpu);
  cudaFree(conflict_col_id_gpu);
  cudaFree(conflict_node_id_gpu);
  cudaFree(existance_of_candidate_rows_gpu);
  cudaFree(conflict_edge_gpu);
  return timer;
}

MeasureTimer Invoke(const ImplVersion version, bool print_result,
                    DataSet *dataset) {
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

MeasureTimer Invoke(const ImplVersion version, bool print_result,
                    DataSets *datasets) {
  MeasureTimer default_timer;
  switch (version) {
  case ImplVersion::ORIGINAL_GPU_MG:
    return Invoke_ORIGINAL_GPU_MG(datasets, print_result);
  default:
    std::cout << "Not Impl yet" << std::endl;
    return default_timer;
  }
}
