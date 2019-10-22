#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <iostream>

namespace gpu {

template <typename MCSolverTraitsType>
__global__ void delete_rows_and_columns(typename MCSolverTraitsType::MatrixType *dl_matrix, int *deleted_rows,
                                        int *deleted_cols,
                                        const int search_depth,
                                        const int selected_row_id,
                                        const int total_dl_matrix_row_num,
                                        const int total_dl_matrix_col_num) {
  for (int i = threadIdx.x; i < total_dl_matrix_col_num; i = i + blockDim.x) {
    if (dl_matrix[selected_row_id * total_dl_matrix_col_num + i] == 1 &&
        deleted_cols[i] == 0) {
      deleted_cols[i] = search_depth;
      for (int j = 0; j < total_dl_matrix_row_num; j++) {
        if (dl_matrix[j * total_dl_matrix_col_num + i] == 1 &&
            deleted_rows[j] == 0) {
          atomicExch(deleted_rows + j, search_depth);
        }
      }
    }
  }
}

template <typename T>
__global__ void init_vectors(T *vec, const int vec_length) {
  for (int i = threadIdx.x; i < vec_length; i = i + blockDim.x) {
    vec[i] = 0;
  }
}

/*
void get_largest_value_launcher(int* vec, cub::KeyValuePair<int, int> *argmax,
int vec_length)
{
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, vec,
argmax, vec_length);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run argmax-reduction
        cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, vec,
argmax, vec_length);
        cudaFree(d_temp_storage);
}
*/

template <typename T>
__global__ void get_largest_value(T *vec, 
                                  const int vec_length, T* max) {
                                 
  for (int i = threadIdx.x; i < vec_length; i = i + blockDim.x) {
    atomicMax(max, vec[i]);
  }

}

template <typename T>
__global__ void find_index(T *vec, const int vec_length, T *value, int *index){
  for (int i = threadIdx.x; i < vec_length; i = i + blockDim.x) {
    if(vec[i]==*value){
      atomicMax(index, i);
    }

  }
}

__global__ void init_vectors_reserved(int *vec, const int vec_length) {
  for (int i = threadIdx.x; i < vec_length; i = i + blockDim.x) {
    if (vec[i] != -1) {
      vec[i] = 0;
    }
  }
}

template <typename MCSolverTraitsType>
__global__ void check_existance_of_candidate_rows(
    int *deleted_rows, typename MCSolverTraitsType::RowGroupType *row_group, const int search_depth, int *token,
    int *selected_row_id, const int total_dl_matrix_row_num) {
  for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
    // std::cout<<deleted_rows[i]<<' '<<row_group[i]<<std::endl;
    if (deleted_rows[i] == 0 && row_group[i] == search_depth) {
      // std::cout<<"Candidate Row Found...."<<std::endl;
      atomicExch(token, 1);
      atomicMin(selected_row_id, i);
    }
  }
  __syncthreads();
}

template <typename MCSolverTraitsType>
__global__ void get_vertex_row_group(typename MCSolverTraitsType::RowGroupType *row_group, typename MCSolverTraitsType::MatrixType *dl_matrix,
                                     const int vertex_num,
                                     const int total_dl_matrix_row_num,
                                     const int total_dl_matrix_col_num) {
  // printf("%d %d\n", vertex_num, total_dl_matrix_row_num);
  for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
    for (int j = 0; j < vertex_num; j++) {
      row_group[i] += dl_matrix[i * total_dl_matrix_col_num + j] * (j + 1);
    }
  }
}

/*
__global__ void select_row(int* deleted_rows, int* row_group, const int
search_depth, const int total_dl_matrix_row_num, int* selected_row_id)
{
        for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i +
blockDim.x)
        {
                if (deleted_rows[i] == 0 && row_group[i] == search_depth)
                {
                        atomicExch(selected_row_id, i);
                        atomicMin(selected_row_id, i);
                }
        }
        __syncthreads();
}
*/

__global__ void recover_deleted_rows(int *deleted_rows, const int search_depth,
                                     const int total_dl_matrix_row_num) {
  for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
    if (abs(deleted_rows[i]) > search_depth ||
        deleted_rows[i] == search_depth) {
      deleted_rows[i] = 0;
    }
  }
}

__global__ void recover_deleted_cols(int *deleted_cols, const int search_depth,
                                     const int total_dl_matrix_col_num) {
  for (int i = threadIdx.x; i < total_dl_matrix_col_num; i = i + blockDim.x) {
    if (deleted_cols[i] >= search_depth) {
      deleted_cols[i] = 0;
    }
  }
}

template <typename MCSolverTraitsType>
__global__ void recover_results(typename MCSolverTraitsType::ResultType *results, const int search_depth,
                                const int total_dl_matrix_row_num) {
  for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
    if (results[i] == search_depth) {
      results[i] = 0;
    }
  }
}

// problem: need to optimized to map on GPU array
template <typename MCSolverTraitsType>
__global__ void get_conflict_node_id(int *deleted_rows, typename MCSolverTraitsType::RowGroupType *row_group,
                                     const int search_depth,
                                     int *conflict_node_id,
                                     const int total_dl_matrix_row_num) {
  for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
    if (row_group[i] == search_depth + 1 &&  deleted_rows[i] < search_depth+1) {
      atomicMax(conflict_node_id, deleted_rows[i]);
    }
  }
  __syncthreads();
}

// problem
template <typename MCSolverTraitsType>
__global__ void get_conflict_edge(typename MCSolverTraitsType::MatrixType *dl_matrix, int *deleted_rows,
                                 int *deleted_cols, typename MCSolverTraitsType::RowGroupType *row_group, 
                                 const int conflict_node_id,
                                 const int search_depth, int *conflict_edge,
                                 const int vertex_num,
                                 const int total_dl_matrix_row_num,
                                 const int total_dl_matrix_col_num) {
  //*conflict_col_id = 0;
  //int idxa = 0;
  //int idxb = 0;

  for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
    // find the conflict edge that connects current node and the most closest
    // node.
    if (deleted_rows[i] == -conflict_node_id) {
      atomicMax(conflict_edge, i);
    } 
    if (row_group[i] == search_depth + 1 &&
               deleted_rows[i] == conflict_node_id) {
      atomicMax(conflict_edge+1, i);
    }
  }
  __syncthreads();
}

template <typename MCSolverTraitsType>
__global__ void get_conflict_col_id(typename MCSolverTraitsType::MatrixType *dl_matrix, int *deleted_cols, 
                                    int *conflict_col_id, int *conflict_edge, 
                                    int total_dl_matrix_col_num, int vertex_num){
  //if(threadIdx.x==0){
  //  printf("conflict edge a %d edge b %d\n",conflict_edge[0],conflict_edge[1]);
 // }
  for (int j = threadIdx.x; j < total_dl_matrix_col_num;
       j = j + blockDim.x) {
    if (dl_matrix[conflict_edge[0] * total_dl_matrix_col_num + j] 
      == dl_matrix[conflict_edge[1] * total_dl_matrix_col_num + j] &&
        deleted_cols[j] > 0 && dl_matrix[conflict_edge[1] * total_dl_matrix_col_num + j]==1) {
      atomicMax(conflict_col_id, j);
    }
  }
  __syncthreads();
}

template <typename MCSolverTraitsType>
__global__ void remove_cols(int *deleted_cols, typename MCSolverTraitsType::ColGroupType *col_group,
                            const int conflict_col_id,
                            const int total_dl_matrix_col_num) {
  for (int i = threadIdx.x; i < total_dl_matrix_col_num; i = i + blockDim.x) {
    if (col_group[i] == col_group[conflict_col_id]) {
      deleted_cols[i] = -1;
    }
  }
}

template <typename T>
__global__ void print_vec(T *vec, int vec_length) {
  for (int i = 0; i < vec_length; i++) {
    printf("%d ", (int)vec[i]);
  }
  printf("\n");
}

template <typename T>
__global__ void print_mat(T *mat[], int total_dl_matrix_row_num,
                          int total_dl_matrix_col_num) {
  for (int i = 0; i < total_dl_matrix_row_num; i++) {
    for (int j = 0; j < total_dl_matrix_col_num; j++) {
      printf("%d ", (int)mat[i][j]);
    }
    printf("\n");
  }
}

template <typename T>
__global__ void add_gpu(T *device_arr, int device_idx, T val) {
  device_arr[device_idx] += val;
  //atomicAdd(&(device_arr[*device_idx]), val);
}

template <typename T>
__global__ void set_vector_value(T *device_var, int idx, T val) {
  device_var[idx] = val;
}




template <typename MCSolverTraitsType>
void mc_solver(typename MCSolverTraitsType::MatrixType *dl_matrix, typename MCSolverTraitsType::ResultType *results, int *deleted_cols,
               int *deleted_rows, typename MCSolverTraitsType::ColGroupType *col_group, typename MCSolverTraitsType::RowGroupType *row_group,
               typename MCSolverTraitsType::ConflictCountType *conflict_count, const int vertex_num_gpu,
               const int total_dl_matrix_row_num_gpu,
               const int total_dl_matrix_col_num_gpu) {
  // to be refreshed if one conflict reaches many counts
  int search_depth = 0;
  int *selected_row_id_gpu;
  int vertex_num = vertex_num_gpu;
  int total_dl_matrix_col_num = total_dl_matrix_col_num_gpu;
  int total_dl_matrix_row_num = total_dl_matrix_row_num_gpu;
  int current_conflict_count;
  int *conflict_node_id_gpu;
  int *conflict_col_id_gpu;
  const int hard_conflict_threshold = 500;
  int *existance_of_candidate_rows_gpu;
  int *existance_of_candidate_rows = new int(0);
  int *conflict_col_id = new int(0);
  int *selected_row_id = new int(0);
  int *conflict_node_id = new int(0);
  int *conflict_edge = new int [2];
  int *conflict_edge_gpu;
  typename MCSolverTraitsType::ConflictCountType *max_gpu;
  cudaMalloc(&existance_of_candidate_rows_gpu, sizeof(int));
  cudaMalloc(&selected_row_id_gpu, sizeof(int));
  cudaMalloc(&conflict_node_id_gpu, sizeof(int));
  cudaMalloc(&conflict_col_id_gpu, sizeof(int));
  cudaMalloc(&max_gpu, sizeof(typename MCSolverTraitsType::ConflictCountType));
  cudaMalloc(&conflict_edge_gpu, sizeof(int)*2);

#ifndef BENCHMARK
  char brk;
#endif

  const int block_count = 1;
  const int thread_count = 32;
  // init lots of vectors
  init_vectors<<<block_count, thread_count>>>(conflict_count,
                                              total_dl_matrix_col_num);
  init_vectors<<<block_count, thread_count>>>(deleted_cols,
                                              total_dl_matrix_col_num);
  init_vectors<<<block_count, thread_count>>>(deleted_rows,
                                              total_dl_matrix_row_num);
  init_vectors<<<block_count, thread_count>>>(results, total_dl_matrix_row_num);
// init_vectors<<<block_count,thread_count>>>(row_group,
// total_dl_matrix_row_num);
//__syncthreads();
// get_vertex_row_group<MCSolverTraitsType><<<block_count,thread_count >>>(row_group, dl_matrix,
// vertex_num, total_dl_matrix_row_num);
//__syncthreads();

// print_mat<<<block_count,thread_count>>>(dl_matrix, total_dl_matrix_row_num,
// total_dl_matrix_col_num);

#ifndef BENCHMARK
  print_vec<<<1, 1>>>(deleted_cols, total_dl_matrix_col_num_gpu);
  cudaDeviceSynchronize();
  print_vec<<<1, 1>>>(deleted_rows, total_dl_matrix_row_num_gpu);
  cudaDeviceSynchronize();
  print_vec<<<1, 1>>>(results, total_dl_matrix_row_num_gpu);
  cudaDeviceSynchronize();
  print_vec<<<1, 1>>>(row_group, total_dl_matrix_row_num_gpu);
  cudaDeviceSynchronize();
  print_vec<<<1, 1>>>(col_group, total_dl_matrix_col_num_gpu);
  cudaDeviceSynchronize();
#endif

  for (search_depth = 1; search_depth <= vertex_num;) {

#ifndef BENCHMARK
    std::cout << "search depth is " << search_depth << std::endl;
    std::cout << "deleted_cols " << std::endl;
    print_vec<<<1, 1>>>(deleted_cols, total_dl_matrix_col_num_gpu);
    cudaDeviceSynchronize();
    std::cout << "deleted_rows " << std::endl;
    print_vec<<<1, 1>>>(deleted_rows, total_dl_matrix_row_num_gpu);
    cudaDeviceSynchronize();
    std::cout << "conflict count " << std::endl;
    print_vec<<<1, 1>>>(conflict_count, total_dl_matrix_col_num_gpu);
    cudaDeviceSynchronize();
    std::cout << "results " << std::endl;
    print_vec<<<1, 1>>>(results, total_dl_matrix_row_num_gpu);
#endif

    cudaDeviceSynchronize();
#ifndef BENCHMARK
    std::cin >> brk;
#endif
    cudaMemset(existance_of_candidate_rows_gpu, 0, sizeof(int));
    cudaMemset(selected_row_id_gpu, 10000, sizeof(int));
    // existance_of_candidate_rows=0;
    // selected_row_id=-1;
    check_existance_of_candidate_rows<MCSolverTraitsType><<<block_count, thread_count>>>(
        deleted_rows, row_group, search_depth, existance_of_candidate_rows_gpu,
        selected_row_id_gpu, total_dl_matrix_row_num);
    //__syncthreads();
    cudaMemcpy(existance_of_candidate_rows, existance_of_candidate_rows_gpu,
               sizeof(int), cudaMemcpyDeviceToHost);

#ifndef BENCHMARK
    std::cout << "check_existance_of_candidate_rows " << *existance_of_candidate_rows <<std::endl;
#endif
    if (*existance_of_candidate_rows ==
        1) { // check if there are candidate rows existing, if no, do backtrace
      // select_row <<<block_count, thread_count >>> (deleted_rows, row_group,
      // search_depth, total_dl_matrix_row_num, selected_row_id_gpu); //select
      // row and add to results
      cudaMemcpy(selected_row_id, selected_row_id_gpu, sizeof(int),
                 cudaMemcpyDeviceToHost);

#ifndef BENCHMARK
      std::cout << "selected row id is " << *selected_row_id << std::endl;
#endif
      //__syncthreads();
      // cudaMemset(&results[*selected_row_id],search_depth,sizeof(int));
      set_vector_value<typename std::remove_pointer<decltype(results)>::type><<<1, 1>>>(results, *selected_row_id, search_depth);
      delete_rows_and_columns<MCSolverTraitsType><<<block_count, thread_count>>>(
          dl_matrix, deleted_rows, deleted_cols, search_depth, *selected_row_id,
          total_dl_matrix_row_num,
          total_dl_matrix_col_num); // delete covered rows and columns
      //__syncthreads();
      // deleted_rows[*selected_row_id] = -search_depth;
      set_vector_value<typename std::remove_pointer<decltype(deleted_rows)>::type><<<1, 1>>>(deleted_rows, *selected_row_id, -search_depth);

      search_depth++; // next step
      // print_vec(deleted_cols, total_dl_matrix_col_num);
      // print_vec(deleted_rows, total_dl_matrix_row_num);
      // print_vec(conflict_count, total_dl_matrix_col_num);
      // print_vec(results, total_dl_matrix_row_num);
      continue;
    } else { // do backtrace
      cudaMemset(conflict_node_id_gpu, 0, sizeof(int));
      cudaMemset(conflict_col_id_gpu, 0, sizeof(int));
      init_vectors<<<1, 2>>>(conflict_edge_gpu, 2);
#ifndef BENCHMARK
      std::cout<<"search depth = "<< search_depth << std::endl;
#endif
      search_depth--;
#ifndef BENCHMARK
      std::cout<<"search depth = "<< search_depth << std::endl;
#endif
      if (search_depth > 0) {
        // conflict_node_id = get_conflict_node_id<MCSolverTraitsType>(deleted_rows, row_group,
        // search_depth, total_dl_matrix_row_num);
        get_conflict_node_id<MCSolverTraitsType><<<block_count, thread_count>>>(
            deleted_rows, row_group, search_depth, conflict_node_id_gpu,
            total_dl_matrix_row_num);
        cudaMemcpy(conflict_node_id, conflict_node_id_gpu, sizeof(int),
                   cudaMemcpyDeviceToHost);

        if(*conflict_node_id>0){
          get_conflict_edge<MCSolverTraitsType><<<block_count, thread_count>>>(
              dl_matrix, deleted_rows, deleted_cols, row_group, *conflict_node_id,
              search_depth, conflict_edge_gpu, vertex_num,
              total_dl_matrix_row_num, total_dl_matrix_col_num);
          
          cudaMemcpy(conflict_edge, conflict_edge_gpu, sizeof(int)*2,
              cudaMemcpyDeviceToHost);

          get_conflict_col_id<MCSolverTraitsType><<<block_count, thread_count>>>(
              dl_matrix, deleted_cols, conflict_col_id_gpu, 
              conflict_edge_gpu, total_dl_matrix_col_num, vertex_num);

          cudaMemcpy(conflict_col_id, conflict_col_id_gpu, sizeof(int),
                    cudaMemcpyDeviceToHost);
          
#ifndef BENCHMARK
          std::cout<<"conflict node id is "<<*conflict_node_id<<std::endl;

          std::cout<<"conflict col id is "<<*conflict_col_id<<std::endl;
          if(*conflict_col_id==0){
            std::cout<<"conflict edge a is "<<conflict_edge[0]<<std::endl;
            std::cout<<"conflict edge b is "<<conflict_edge[1]<<std::endl;
            cudaDeviceSynchronize();
            std::cout << "row 1 " << std::endl;
            print_vec<<<1, 1>>>(dl_matrix+conflict_edge[0]*total_dl_matrix_col_num_gpu, total_dl_matrix_col_num_gpu);
            cudaDeviceSynchronize();
            std::cout << "row 2 " << std::endl;
            print_vec<<<1, 1>>>(dl_matrix+conflict_edge[1]*total_dl_matrix_col_num_gpu, total_dl_matrix_col_num_gpu);
            cudaDeviceSynchronize();
          }
#endif

          // conflict_count[*conflict_col_id]++; //update conflict edge count
          add_gpu<typename std::remove_pointer<decltype(conflict_count)>::type><<<1, 1>>>(conflict_count, *conflict_col_id, 1);
          recover_deleted_rows<<<block_count, thread_count>>>(
              deleted_rows, search_depth,
              total_dl_matrix_row_num); // recover deleted rows  previously
                                        // selected rows
          //__syncthreads();
          recover_deleted_cols<<<block_count, thread_count>>>(
              deleted_cols, search_depth,
              total_dl_matrix_col_num); // recover deleted cols except afftected
                                        // by previously selected rows
          //__syncthreads();
          recover_results<MCSolverTraitsType><<<block_count, thread_count>>>(
              results, search_depth, total_dl_matrix_row_num); // recover results
          //__syncthreads();
          cudaMemcpy(&current_conflict_count, &conflict_count[*conflict_col_id],
                    sizeof(int), cudaMemcpyDeviceToHost);
          if (current_conflict_count > hard_conflict_threshold) {
            search_depth = 1;
            init_vectors<<<block_count, thread_count>>>(conflict_count,
                                                        total_dl_matrix_col_num);
            init_vectors_reserved<<<block_count, thread_count>>>(
                deleted_cols, total_dl_matrix_col_num);
            init_vectors<<<block_count, thread_count>>>(deleted_rows,
                                                        total_dl_matrix_row_num);
            init_vectors<<<block_count, thread_count>>>(results,
                                                        total_dl_matrix_row_num);
            //__syncthreads();
            remove_cols<MCSolverTraitsType><<<block_count, thread_count>>>(deleted_cols, col_group,
                                                      *conflict_col_id,
                                                      total_dl_matrix_col_num);

            //__syncthreads();
            // deleted_cols[*conflict_col_id] = -1;
            cudaMemset(&deleted_cols[*conflict_col_id], -1, sizeof(int));
          }
        } else {
          recover_deleted_rows<<<block_count, thread_count>>>(
            deleted_rows, search_depth,
            total_dl_matrix_row_num); // recover deleted rows  previously
                                      // selected rows
          //__syncthreads();
          recover_deleted_cols<<<block_count, thread_count>>>(
              deleted_cols, search_depth,
              total_dl_matrix_col_num); // recover deleted cols except afftected
                                        // by previously selected rows
          //__syncthreads();
          recover_results<MCSolverTraitsType><<<block_count, thread_count>>>(
              results, search_depth, total_dl_matrix_row_num); // recover results
          //__syncthreads();
        }
      } else { // if all vertices are gone through, directly remove the edge
               // with largest conflict count.
#ifndef BENCHMARK
        std::cout<<"reset state"<<std::endl;
        std::cout<<"======================================================================================"<<std::endl;
#endif
        search_depth = 1;
        get_largest_value<typename std::remove_pointer<decltype(conflict_count)>::type><<<block_count, thread_count>>>(
            conflict_count, total_dl_matrix_col_num, max_gpu);
        find_index<typename std::remove_pointer<decltype(conflict_count)>::type><<<block_count, thread_count>>>(
            conflict_count, total_dl_matrix_col_num, max_gpu, conflict_col_id_gpu
        );
        cudaMemcpy(conflict_col_id, conflict_col_id_gpu, sizeof(int),
                   cudaMemcpyDeviceToHost);
        //__syncthreads();
        init_vectors<<<block_count, thread_count>>>(conflict_count,
                                                    total_dl_matrix_col_num);
        init_vectors_reserved<<<block_count, thread_count>>>(
            deleted_cols, total_dl_matrix_col_num);
        init_vectors<<<block_count, thread_count>>>(deleted_rows,
                                                    total_dl_matrix_row_num);
        init_vectors<<<block_count, thread_count>>>(results,
                                                    total_dl_matrix_row_num);
        //__syncthreads();
        remove_cols<MCSolverTraitsType><<<block_count, thread_count>>>(
            deleted_cols, col_group, *conflict_col_id, total_dl_matrix_col_num);
        continue;
      }
      // print_vec(deleted_cols, total_dl_matrix_col_num);
      // print_vec(deleted_rows, total_dl_matrix_row_num);
      // print_vec(conflict_count, total_dl_matrix_col_num);
      // print_vec(results, total_dl_matrix_row_num);
    }
  }

  cudaFree(existance_of_candidate_rows_gpu);
  cudaFree(selected_row_id_gpu);
  cudaFree(conflict_col_id_gpu);
  cudaFree(conflict_node_id_gpu);
  cudaFree(conflict_edge);
  delete existance_of_candidate_rows;
  delete conflict_col_id;
  delete selected_row_id;
  delete conflict_node_id;
  delete [] conflict_edge;
}

} // namespace gpu
