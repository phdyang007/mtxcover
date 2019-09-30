#include "MatrixCoverGPU.cuh"

namespace gpu_mg {

constexpr int size_bit = 1 << 31;

__device__ void delete_rows_and_columns(
    const int *dl_matrix, const int *next_row, const int *next_col,
    short *deleted_rows, short *deleted_cols, const int search_depth,
    const int selected_row_id, const int total_dl_matrix_row_num,
    const int total_dl_matrix_col_num) {
  int selected_row_idx = selected_row_id * total_dl_matrix_col_num;

  for (int i = threadIdx.x; i < total_dl_matrix_col_num;
       // // The below line will have negative effect of the col number is small
       //  i += (next_col[selected_row_idx + i] + blockDim.x - 1) / blockDim.x
       i += blockDim.x) {
    if (deleted_cols[i] == 0 && dl_matrix[selected_row_idx + i] == 1) {
      deleted_cols[i] = search_depth;
      for (int j = 0; j < total_dl_matrix_row_num;
           j += next_row[i * total_dl_matrix_row_num + j]) {
        if (deleted_rows[j] == 0 &&
            dl_matrix[j * total_dl_matrix_col_num + i] == 1) {
          deleted_rows[j] = search_depth;
        }
      }
    }
  }
  __syncthreads();
}

template <typename T> 
__device__ void init_vectors(T *vec, const int vec_length) {
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
__device__ void get_largest_value(T *vec, const int vec_length, int *max) {

  for (int i = threadIdx.x; i < vec_length; i = i + blockDim.x) {

    atomicMax(max, vec[i]);
  }
}


template <typename T>
__device__ void find_index(T *vec, const int vec_length, int *value,
                           int *index) {
  for (int i = threadIdx.x; i < vec_length; i = i + blockDim.x) {
    if (vec[i] == *value) {
      atomicMax(index, i);
    }
  }
}


template <typename T>
__device__ void init_vectors_reserved(T *vec, const int vec_length) {
  for (int i = threadIdx.x; i < vec_length; i = i + blockDim.x) {
    // if (vec[i] != -1) {
    vec[i] &= size_bit;
    // }
  }
}

__device__ void check_existance_of_candidate_rows(
    short *deleted_rows, const int *row_group, int search_depth, int *token,
    int *selected_row_id, const int total_dl_matrix_row_num) {
  for (int i = threadIdx.x; i<total_dl_matrix_row_num && * selected_row_id> i;
       i = i + blockDim.x) {
    // std::cout<<deleted_rows[i]<<' '<<row_group[i]<<std::endl;
    if (deleted_rows[i] == 0 && row_group[i] == search_depth) {
      // std::cout<<"Candidate Row Found...."<<std::endl;
      // atomicExch(token, 1);
      *token = 1;
      atomicMin(selected_row_id, i);
      // If find a number can break;
      break;
    }
  }
  __syncthreads();
}

__device__ void get_vertex_row_group(int *row_group, int *dl_matrix,
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
__device__ void select_row(int* deleted_rows, int* row_group, const int
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

__device__ void recover_deleted_rows(short *deleted_rows, const int search_depth,
                                     const int total_dl_matrix_row_num) {
  for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
    if (abs(deleted_rows[i]) > search_depth ||
        deleted_rows[i] == search_depth) {
      deleted_rows[i] = 0;
    }
  }
}

__device__ void recover_deleted_cols(short *deleted_cols, const int search_depth,
                                     const int total_dl_matrix_col_num) {
  for (int i = threadIdx.x; i < total_dl_matrix_col_num; i = i + blockDim.x) {
    if (deleted_cols[i] >= search_depth) {
      deleted_cols[i] = 0;
    }
  }
}

__device__ void recover_results(short *results, const int search_depth,
                                const int total_dl_matrix_row_num) {
  for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
    if (results[i] == search_depth) {
      results[i] = 0;
    }
  }
}

// problem: need to optimized to map on GPU array
__device__ void get_conflict_node_id(short *deleted_rows, int *row_group,
                                     const int search_depth,
                                     int *conflict_node_id,
                                     const int total_dl_matrix_row_num) {
  for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
    if (row_group[i] == search_depth + 1 &&
        deleted_rows[i] < search_depth + 1) {
      atomicMax(conflict_node_id, deleted_rows[i]);
    }
  }
  __syncthreads();
}

__device__ void get_conflict_edge(const int *dl_matrix, short *deleted_rows,
                                  const int *row_group, int conflict_node_id,
                                  int search_depth, int *conflict_edge,
                                  const int vertex_num,
                                  const int total_dl_matrix_row_num,
                                  const int total_dl_matrix_col_num) {
  //*conflict_col_id = 0;
  // int idxa = 0;
  // int idxb = 0;

  for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
    // find the conflict edge that connects current node and the most closest
    // node.
    if (deleted_rows[i] == -conflict_node_id) {
      atomicMax(conflict_edge, i);
    }
    if (row_group[i] == search_depth + 1 &&
        deleted_rows[i] == conflict_node_id) {
      atomicMax(conflict_edge + 1, i);
    }
  }
  __syncthreads();
}

__device__ void get_conflict_col_id(const int *dl_matrix, short *deleted_cols,
                                    int *conflict_col_id, int *conflict_edge,
                                    int total_dl_matrix_col_num,
                                    int vertex_num) {
  // if(threadIdx.x==0){
  //  printf("conflict edge a %d edge b
  //  %d\n",conflict_edge[0],conflict_edge[1]);
  // }
  for (int j = threadIdx.x; j < total_dl_matrix_col_num; j = j + blockDim.x) {
    if (dl_matrix[conflict_edge[0] * total_dl_matrix_col_num + j] ==
            dl_matrix[conflict_edge[1] * total_dl_matrix_col_num + j] &&
        deleted_cols[j] > 0 &&
        dl_matrix[conflict_edge[1] * total_dl_matrix_col_num + j] == 1) {
      atomicMax(conflict_col_id, j);
    }
  }
  __syncthreads();
}

__device__ void remove_cols(short *deleted_cols, const int *col_group,
                            int conflict_col_id,
                            const int total_dl_matrix_col_num) {
  for (int i = threadIdx.x; i < total_dl_matrix_col_num; i = i + blockDim.x) {
    if (col_group[i] == col_group[conflict_col_id]) {
      deleted_cols[i] = size_bit;
    }
  }
}

__device__ void print_vec(int *vec, int vec_length) {
  for (int i = 0; i < vec_length; i++) {
    printf("%d ", vec[i]);
  }
  printf("\n");
}

/*
__global__ inline void print_vec_g(int *vec, int vec_length)
{
        for(int i=0; i<vec_length; i++)
        {
                printf("%d ", vec[i]);
        }
        printf("\n");
}

*/
__device__ void print_mat(int *mat[], int total_dl_matrix_row_num,
                          int total_dl_matrix_col_num) {
  for (int i = 0; i < total_dl_matrix_row_num; i++) {
    for (int j = 0; j < total_dl_matrix_col_num; j++) {
      printf("%d ", mat[i][j]);
    }
    printf("\n");
  }
}

__device__ void add_gpu(int *device_var, int val) {
  atomicAdd(device_var, val);
}

__device__ void set_vector_value(int *device_var, int idx, int val) {
  device_var[idx] = val;
}

__global__ void 
init_vertex_group(int *row_group, int *dl_matrix, int* vertex_num, int* t_cn, int* t_rn, int *offset_row, int *offset_matrix, int graph_count) {
    int k=blockIdx.x;
    if(k<graph_count){
        get_vertex_row_group(row_group+offset_row[k], dl_matrix+offset_matrix[k], vertex_num[k], t_rn[k], t_cn[k]);
    }

}

__global__ void
mc_solver(int *dl_matrix, int *next_col, int *next_row, int *all_results,
          int *all_deleted_cols, int *all_deleted_rows, 
          int *col_group, int *row_group,
          int *all_conflict_count, int *vertex_num, int *total_dl_matrix_row_num,
          int *total_dl_matrix_col_num, int *offset_col, int *offset_row,
          int *offset_matrix, int *all_search_depth, int *all_selected_row_id,
          int *current_conflict_count, int *all_conflict_node_id,
          int *all_conflict_col_id, int *all_existance_of_candidate_rows,
          int *all_conflict_edge, int *all_max,const int graph_count,
          const int hard_conflict_threshold) {

  // to be refreshed if one conflict reaches many counts
  /*
  int search_depth = 0;
  int *selected_row_id_gpu;
  int vertex_num = vertex_num_gpu;
  int total_dl_matrix_col_num=total_dl_matrix_col_num_gpu;
  int total_dl_matrix_row_num=total_dl_matrix_row_num_gpu;
  int current_conflict_count;
  int *conflict_node_id_gpu;
  int *conflict_col_id_gpu;
  const int hard_conflict_threshold = 500;
  int *existance_of_candidate_rows_gpu;
  int *existance_of_candidate_rows=new int(0);
  int *conflict_col_id=new int(0);
  int *selected_row_id=new int(0);
  int *conflict_node_id=new int(0);
  cudaMalloc(&existance_of_candidate_rows_gpu, sizeof(int));
  cudaMalloc(&selected_row_id_gpu, sizeof(int));
  cudaMalloc(&conflict_node_id_gpu, sizeof(int));
  cudaMalloc(&conflict_col_id_gpu, sizeof(int));

  char brk;
  */
  // int k = blockDim.x;
  __shared__ short deleted_cols[128];
  __shared__ short deleted_rows[256];
  __shared__ short results[256];
  __shared__ unsigned short conflict_count[128];
  __shared__ int conflict_edge[2];
  __shared__ int existance_of_candidate_rows;
  __shared__ int selected_row_id;
  __shared__ int conflict_node_id;
  __shared__ int conflict_col_id;
  __shared__ int search_depth;
  int max = 0;
  int k = blockIdx.x; 
  if (k < graph_count) {
    const int t_cn = total_dl_matrix_col_num[k];
    const int t_rn = total_dl_matrix_row_num[k];
    //int *t_conflict_count = conflict_count + offset_col[k];
    //int *t_deleted_cols = deleted_cols + offset_col[k];
    //int *t_deleted_rows = deleted_rows + offset_row[k];
    int *t_results = all_results + offset_row[k];
    const int *t_row_group = row_group + offset_row[k];
    const int *t_col_group = col_group + offset_col[k];
    const int *t_dl_matrix = dl_matrix + offset_matrix[k];
    const int *t_next_col = next_col + offset_matrix[k];
    const int *t_next_row = next_row + offset_matrix[k];
    //int *t_conflict_edge = conflict_edge + 2 * k;

#ifndef BENCHMARK
    printf("blockID is %d\n", k);
    printf("vertexnum is %d\n", vertex_num[k]);
    printf("init conflict count \n");
#endif
    init_vectors(conflict_count, t_cn);
#ifndef BENCHMARK
    for (int i = 0; i < t_cn; i++) {
      printf("%d ", t_conflict_count[i]);
    }
    printf("\n");
#endif
    init_vectors(deleted_cols, t_cn);
    init_vectors(deleted_rows, t_rn);
    init_vectors(results, t_rn);
    //__syncthreads();
    //get_vertex_row_group(t_row_group, t_dl_matrix, vertex_num[k], t_rn, t_cn);
    __syncthreads();
    /*
    print_vec(deleted_cols+offset_col[k], t_cn);
    __syncthreads();
    print_vec(deleted_rows+offset_row[k], t_rn);
    __syncthreads();
    print_vec(results+offset_row[k], t_rn);
    __syncthreads();
    print_vec(row_group+offset_row[k], t_rn);
    __syncthreads();
    print_vec(col_group+offset_col[k], t_cn);
    __syncthreads();
    */

    for (search_depth = 1; search_depth <= vertex_num[k];) {
#ifndef BENCHMARK
      printf("search depth is %d\n", search_depth[k]);
      // std::cout<<"deleted_cols "<<std::endl;
      // cudaDeviceSynchronize();
      printf("deleted_cols\n");
      print_vec(deleted_cols, t_cn);
      // cudaDeviceSynchronize();
      // cudaDeviceSynchronize();
      printf("deleted_rows\n");
      print_vec(deleted_rows, t_rn);
      // cudaDeviceSynchronize();
      // cudaDeviceSynchronize();
      printf("results\n");
      print_vec(results, t_rn);
// cudaDeviceSynchronize();
#endif

      existance_of_candidate_rows = 0;
      selected_row_id = t_rn;
      conflict_node_id = 0;
      conflict_col_id = 0;
      conflict_edge[0] = 0;
      conflict_edge[1] = 0;
      // existance_of_candidate_rows=0;
      // selected_row_id=-1;
      check_existance_of_candidate_rows(
          deleted_rows, t_row_group, search_depth,
          &existance_of_candidate_rows, &selected_row_id, t_rn);
      __syncthreads();
      // printf()
      // cudaMemcpy(existance_of_candidate_rows,
      // existance_of_candidate_rows_gpu, sizeof(int), cudaMemcpyDeviceToHost);
      // std::cout<<"check_existance_of_candidate_rows "<<std::endl;
      if (existance_of_candidate_rows == 1) { // check if there are candidate
                                                 // rows existing, if no, do
                                                 // backtrace
// select_row <<<block_count, thread_count >>> (deleted_rows, row_group,
// search_depth, total_dl_matrix_row_num, selected_row_id_gpu); //select
// row and add to results
// cudaMemcpy(selected_row_id, selected_row_id_gpu, sizeof(int),
// cudaMemcpyDeviceToHost);
#ifndef BENCHMARK
        printf("selected row id is %d \n", selected_row_id);
#endif
        //__syncthreads();
        // cudaMemset(&results[*selected_row_id],search_depth,sizeof(int));
        results[selected_row_id] = search_depth;
        // set_vector_value<<<1,1>>>(results, *selected_row_id, search_depth);
        delete_rows_and_columns(t_dl_matrix, t_next_row, t_next_col,
                                deleted_rows, deleted_cols, search_depth,
                                selected_row_id, t_rn,
                                t_cn); // delete covered rows and columns
        __syncthreads();
        // deleted_rows[*selected_row_id] = -search_depth;
        deleted_rows[selected_row_id] = -search_depth;
        // set_vector_value<<<1,1>>>(deleted_rows, *selected_row_id,
        // -search_depth);

        search_depth++; // next step
        // print_vec(deleted_cols, total_dl_matrix_col_num);
        // print_vec(deleted_rows, total_dl_matrix_row_num);
        // print_vec(conflict_count, total_dl_matrix_col_num);
        // print_vec(results, total_dl_matrix_row_num);
      } else { // do backtrace
        search_depth--;
        if (search_depth > 0) {

          get_conflict_node_id(deleted_rows, row_group, search_depth,
                               &conflict_node_id, t_rn);
          if (conflict_node_id > 0) {
            __syncthreads();
            get_conflict_edge(t_dl_matrix, deleted_rows, t_row_group,
                              conflict_node_id, search_depth,
                              conflict_edge, vertex_num[k], t_rn, t_cn);
            __syncthreads();
            get_conflict_col_id(t_dl_matrix, deleted_cols,
                                &conflict_col_id, conflict_edge, t_cn,
                                vertex_num[k]);
            __syncthreads();

            // conflict_count[*conflict_col_id]++;
            // update conflict edge count
            conflict_count[conflict_col_id]++;
            // add_gpu<<<1,1>>>(&deleted_rows[*selected_row_id],1);
            recover_deleted_rows(deleted_rows, search_depth,
                                 t_rn); // recover deleted
                                        // rows  previously
                                        // selected rows
            __syncthreads();
            recover_deleted_cols(deleted_cols, search_depth,
                                 t_cn); // recover deleted
                                        // cols except
                                        // afftected by
                                        // previously
                                        // selected rows
            __syncthreads();
            recover_results(results, search_depth,
                            t_rn); // recover results
            //__syncthreads();
            // cudaMemcpy(&current_conflict_count,
            // &conflict_count[*conflict_col_id], sizeof(int),
            // cudaMemcpyDeviceToHost);
            if (conflict_count[conflict_col_id] >
                hard_conflict_threshold) {
              search_depth = 1;
              init_vectors(conflict_count, t_cn);
              init_vectors_reserved(deleted_cols, t_cn);
              init_vectors(deleted_rows, t_rn);
              init_vectors(results, t_rn);
              __syncthreads();
              remove_cols(deleted_cols, col_group, conflict_col_id,
                          t_cn);
              __syncthreads();
              deleted_cols[conflict_col_id] = size_bit;

              // /cudaMemset(&deleted_cols[*conflict_col_id],-1,sizeof(int));
            }
          } else {
            recover_deleted_rows(deleted_rows, search_depth,
                                 t_rn); // recover deleted
                                        // rows  previously
                                        // selected rows
            __syncthreads();
            recover_deleted_cols(deleted_cols, search_depth,
                                 t_cn); // recover deleted
                                        // cols except
                                        // afftected by
                                        // previously
                                        // selected rows
            __syncthreads();
            recover_results(results, search_depth,
                            t_rn); // recover results
          }
        } else { // if all vertices are gone through, directly remove the edge
                 // with largest conflict count.
          search_depth = 1;
          max = 0;
          get_largest_value(conflict_count, t_cn, &max);

          // cudaMemcpy(conflict_col_id, conflict_col_id_gpu, sizeof(int),
          // cudaMemcpyDeviceToHost);
          __syncthreads();
          find_index(conflict_count, t_cn, &max, &conflict_col_id);
          init_vectors(conflict_count, t_cn);
          init_vectors_reserved(deleted_cols, t_cn);
          init_vectors(deleted_rows, t_rn);
          init_vectors(results, t_rn);
          __syncthreads();
          remove_cols(deleted_cols, t_col_group, conflict_col_id, t_cn);
        }
        // print_vec(deleted_cols, total_dl_matrix_col_num);
        // print_vec(deleted_rows, total_dl_matrix_row_num);
        // print_vec(conflict_count, total_dl_matrix_col_num);
        // print_vec(results, total_dl_matrix_row_num);
      }
    }
    __syncthreads();
    for(int i=threadIdx.x; i < t_rn; i+=blockDim.x){
        t_results[i]=results[i];
        printf("result is %d", results[i]);
    }
  }
}

} // namespace gpu_mg
