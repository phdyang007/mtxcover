#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <iostream>

//#include "cub/cub/cub.cuh"

namespace gpu_mg {

//constexpr int size_bit = 1 << 31;

template <typename MCSolverTraitsType>
__device__ void delete_rows_and_columns(
    typename MCSolverTraitsType::MatrixType *dl_matrix, const typename MCSolverTraitsType::NextRowType *next_row, typename MCSolverTraitsType::NextColType *next_col,
    short *deleted_rows, short *deleted_cols, const int search_depth,
    const int selected_row_id, const int total_dl_matrix_row_num,
    const int total_dl_matrix_col_num) {
  auto selected_row = dl_matrix+selected_row_id * total_dl_matrix_col_num;
  ///*
  for (int i = threadIdx.x; i < total_dl_matrix_col_num;
       // // The below line will have negative effect of the col number is small
       //  i += (next_col[selected_row_idx + i] + blockDim.x - 1) / blockDim.x
       i += blockDim.x) {
    if (deleted_cols[i] == 0 && selected_row[i] == 1) {
      deleted_cols[i] = search_depth;
      //atomicInc(&tmp_deleted_cols_count)
      for (int j = 0; j < total_dl_matrix_row_num;
           j += next_row[i * total_dl_matrix_row_num + j]) {
        if (deleted_rows[j] == 0 &&
            dl_matrix[j * total_dl_matrix_col_num + i] == 1) {
          deleted_rows[j] = search_depth;

        }
      }
    }    
  }
  //*/
  /*
  int * tmp_row;
  int * tmp_next_col;
  for (int i = threadIdx.x; i < total_dl_matrix_row_num; i += blockDim.x){
    tmp_row = dl_matrix + i * total_dl_matrix_col_num;
    tmp_next_col = next_col + i * total_dl_matrix_col_num;
    for (int j = 0; j < total_dl_matrix_col_num; j += tmp_next_col[j]){
      if (tmp_row[j] + selected_row[j] == 2 && deleted_cols[j] !=-1){
        deleted_rows[i] = deleted_rows[i]==0?search_depth:deleted_rows[i];
        deleted_cols[j] = deleted_cols[j]==0?search_depth:deleted_cols[j];
      }
    }
  }
  */

}

__device__ void init_vectors(short *vec, const int vec_length) {
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

__device__ void get_largest_value(short *vec, const int vec_length, int *max) {

  for (int i = threadIdx.x; i < vec_length; i = i + blockDim.x) {

    atomicMax(max, vec[i]);
  }
}

__device__ void find_index(short *vec, const int vec_length, int *value,
                           int *index) {
  for (int i = threadIdx.x; i < vec_length; i = i + blockDim.x) {
    if (vec[i] == *value) {
      atomicMax(index, i);
    }
  }
}

__device__ void init_vectors_reserved(short *vec, const int vec_length) {
  for (int i = threadIdx.x; i < vec_length; i = i + blockDim.x) {
    if (vec[i] != -1) {
      vec[i] = 0;
    }
  }
}

template <typename MCSolverTraitsType>
__device__ void check_existance_of_candidate_rows(
    short *deleted_rows, typename MCSolverTraitsType::RowGroupType *row_group, const int search_depth, int *token,
    int *selected_row_id, const int total_dl_matrix_row_num) {
  for (int i = threadIdx.x; i<total_dl_matrix_row_num;
       i = i + blockDim.x) {
    // std::cout<<deleted_rows[i]<<' '<<row_group[i]<<std::endl;
    if (deleted_rows[i] == 0 && row_group[i] == search_depth) {
      // std::cout<<"Candidate Row Found...."<<std::endl;
      atomicExch(token, 1);
      atomicMin(selected_row_id, i);
      // If find a number can break;
      //break;
    }
  }
}

template <typename MCSolverTraitsType>
__device__ void get_vertex_row_group(typename MCSolverTraitsType::RowGroupType *row_group, typename MCSolverTraitsType::MatrixType *dl_matrix,
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
template <typename MCSolverTraitsType>
__device__ void get_conflict_node_id(short *deleted_rows, typename MCSolverTraitsType::RowGroupType *row_group,
                                     const int search_depth,
                                     int *conflict_node_id,
                                     const int total_dl_matrix_row_num) {
  for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
    if (row_group[i] == search_depth + 1 &&
        deleted_rows[i] < search_depth + 1) {
      atomicMax(conflict_node_id, deleted_rows[i]);
    }
  }
  
}

template <typename MCSolverTraitsType>
__device__ void get_conflict_edge(typename MCSolverTraitsType::MatrixType *dl_matrix, short *deleted_rows,
                                  typename MCSolverTraitsType::RowGroupType *row_group, const int conflict_node_id,
                                  const int search_depth, int *conflict_edge,
                                  const int vertex_num,
                                  const int total_dl_matrix_row_num,
                                  const int total_dl_matrix_col_num) {
  //*conflict_col_id = 0;
  // int idxa = 0;
  // int idxb = 0;

  for (int i = threadIdx.x; i < total_dl_matrix_row_num; i += blockDim.x) {
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

}

template <typename MCSolverTraitsType>
__device__ void get_conflict_col_id(typename MCSolverTraitsType::MatrixType *dl_matrix, short *deleted_cols,
                                    int *conflict_col_id, int *conflict_edge,
                                    int total_dl_matrix_col_num,
                                    int vertex_num) {
  // if(threadIdx.x==0){
  //  printf("conflict edge a %d edge b
  //  %d\n",conflict_edge[0],conflict_edge[1]);
  // }
  auto edge_a_dlmatrix = dl_matrix+conflict_edge[0] * total_dl_matrix_col_num;
  auto edge_b_dlmatrix = dl_matrix+conflict_edge[1] * total_dl_matrix_col_num; 
  for (int j = threadIdx.x; j < total_dl_matrix_col_num; j += blockDim.x) {
    if (edge_a_dlmatrix[j] == edge_b_dlmatrix[j] &&
        deleted_cols[j] > 0 && edge_b_dlmatrix[j] == 1) {
      atomicMax(conflict_col_id, j);
    }
  }

}

template <typename MCSolverTraitsType>
__device__ void remove_cols(short *deleted_cols, typename MCSolverTraitsType::ColGroupType *col_group,
                            const int conflict_col_id,
                            const int total_dl_matrix_col_num) {
  for (int i = threadIdx.x; i < total_dl_matrix_col_num; i = i + blockDim.x) {
    if (col_group[i] == col_group[conflict_col_id]) {
      deleted_cols[i] = -1;
    }
  }
}

template <typename T>
__device__ void print_vec(T *vec, int vec_length) {
  for (int i = 0; i < vec_length; i++) {
    printf("%d ", (int)vec[i]);
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

template <typename T>
__device__ void print_mat(T *mat[], int total_dl_matrix_row_num,
                          int total_dl_matrix_col_num) {
  for (int i = 0; i < total_dl_matrix_row_num; i++) {
    for (int j = 0; j < total_dl_matrix_col_num; j++) {
      printf("%d ", (int)mat[i][j]);
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

template <typename MCSolverTraitsType>
__global__ void 
init_vertex_group(typename MCSolverTraitsType::RowGroupType *row_group, typename MCSolverTraitsType::MatrixType *dl_matrix, int* vertex_num, int* t_cn, int* t_rn, int *offset_row, int *offset_matrix, int graph_count) {
    int k=blockIdx.x;
    if(k<graph_count){
        get_vertex_row_group<MCSolverTraitsType>(row_group+offset_row[k], dl_matrix+offset_matrix[k], vertex_num[k], t_rn[k], t_cn[k]);
    }

}

///
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
#define GRAPH_PER_BLOCK 1



template <typename MCSolverTraitsType>
__global__ void
mc_solver(typename MCSolverTraitsType::MatrixType *dl_matrix, typename MCSolverTraitsType::NextColType *next_col, typename MCSolverTraitsType::NextRowType *next_row, typename MCSolverTraitsType::ResultType *results,
          int *_deleted_cols, int *_deleted_rows, typename MCSolverTraitsType::ColGroupType *col_group, typename MCSolverTraitsType::RowGroupType *row_group,
          typename MCSolverTraitsType::ConflictCountType *conflict_count, int *vertex_num, int *total_dl_matrix_row_num,
          int *total_dl_matrix_col_num, int *offset_col, int *offset_row,
          int *offset_matrix, int *_search_depth, int *_selected_row_id,
          int *_current_conflict_count, int *_conflict_node_id,
          int *_conflict_col_id, int *_existance_of_candidate_rows,
          int *_conflict_edge, int *_max, const int graph_count,
          const int hard_conflict_threshold, const int graph_per_block) 
{

    //add shared mem

    __shared__ short t_deleted_rows[GRAPH_PER_BLOCK][MCSolverTraitsType::max_num_rows];
    __shared__ short t_deleted_cols[GRAPH_PER_BLOCK][MCSolverTraitsType::max_num_cols];
    __shared__ short t_conflict_count[GRAPH_PER_BLOCK][MCSolverTraitsType::max_num_cols];
    __shared__ short t_results[GRAPH_PER_BLOCK][MCSolverTraitsType::max_num_rows];
    __shared__ int t_conflict_edge[GRAPH_PER_BLOCK][2];
    __shared__ short search_depth[GRAPH_PER_BLOCK];
    __shared__ int t_max[GRAPH_PER_BLOCK];
    __shared__ int t_existance_of_candidate_rows[GRAPH_PER_BLOCK];
    __shared__ int t_conflict_node_id[GRAPH_PER_BLOCK];
    __shared__ int t_conflict_col_id[GRAPH_PER_BLOCK];
    __shared__ int t_vertex_num[GRAPH_PER_BLOCK];
    __shared__ int t_selected_row_id[GRAPH_PER_BLOCK];
    //
    __shared__ int t_cn[GRAPH_PER_BLOCK];
    __shared__ int t_rn[GRAPH_PER_BLOCK];
    __shared__ typename MCSolverTraitsType::ResultType *t_final_results[GRAPH_PER_BLOCK];
    __shared__ typename MCSolverTraitsType::RowGroupType *t_row_group[GRAPH_PER_BLOCK];
    __shared__ typename MCSolverTraitsType::ColGroupType *t_col_group[GRAPH_PER_BLOCK];
    __shared__ typename MCSolverTraitsType::MatrixType *t_dl_matrix[GRAPH_PER_BLOCK];
    __shared__ typename MCSolverTraitsType::NextColType *t_next_col[GRAPH_PER_BLOCK];
    __shared__ typename MCSolverTraitsType::NextRowType *t_next_row[GRAPH_PER_BLOCK];
    //__shared__ int delete_rows_and_columns_count; 

    //end add shared mem
    int sub_graph_id = threadIdx.y;
    int k = blockIdx.x * graph_per_block + sub_graph_id;

    //int k_p = threadIdx.y;
    if (k < graph_count /*&& (k == 1784 || k == 1778 || k == 1792 || k == 2352 || k == 2350 || k == 2351 || k == 2349)*/)
    {
        if (threadIdx.x == 0)
        {
            t_cn[sub_graph_id] = total_dl_matrix_col_num[k];
            t_rn[sub_graph_id] = total_dl_matrix_row_num[k];
            //int *t_conflict_count[sub_graph_id] = conflict_count + offset_col[k];
            //int *t_deleted_cols[sub_graph_id] = deleted_cols + offset_col[k];
            //int *t_deleted_rows[sub_graph_id] = deleted_rows + offset_row[k];
            t_final_results[sub_graph_id]  = results + offset_row[k];
            t_row_group[sub_graph_id]      = row_group + offset_row[k];
            t_col_group[sub_graph_id] = col_group + offset_col[k];
            t_dl_matrix[sub_graph_id] = dl_matrix + offset_matrix[k];
            t_next_col[sub_graph_id] = next_col + offset_matrix[k];
            t_next_row[sub_graph_id]= next_row + offset_matrix[k];

            t_vertex_num[sub_graph_id] = vertex_num[k];

            search_depth[sub_graph_id] = 1; 
            //delete_rows_and_columns_count = 0; 
        }
        __syncthreads();
        //int *t_conflict_edge[sub_graph_id] = conflict_edge + 2 * k;


#ifndef BENCHMARK
        printf("blockID is %d\n", k);
        printf("vertexnum is %d\n", vertex_num[k]);
        printf("init conflict count \n");
#endif
        init_vectors(t_conflict_count[sub_graph_id], t_cn[sub_graph_id]);
#ifndef BENCHMARK
        for (int i = 0; i < t_cn[sub_graph_id]; i++) {
            printf("%d ", t_conflict_count[sub_graph_id][i]);
        }
        printf("\n");
#endif
        init_vectors(t_deleted_cols[sub_graph_id], t_cn[sub_graph_id]);
        init_vectors(t_deleted_rows[sub_graph_id], t_rn[sub_graph_id]);
        init_vectors(t_results[sub_graph_id], t_rn[sub_graph_id]);
        __syncthreads();

        //get_vertex_row_group<MCSolverTraitsType>(t_row_group[sub_graph_id], t_dl_matrix[sub_graph_id], vertex_num[k], t_rn[sub_graph_id], t_cn[sub_graph_id]);
        //__syncthreads();
        /*
           print_vec(deleted_cols+offset_col[k], t_cn[sub_graph_id]);
           __syncthreads();
           print_vec(deleted_rows+offset_row[k], t_rn[sub_graph_id]);
           __syncthreads();
           print_vec(results+offset_row[k], t_rn[sub_graph_id]);
           __syncthreads();
           print_vec(row_group+offset_row[k], t_rn[sub_graph_id]);
           __syncthreads();
           print_vec(col_group+offset_col[k], t_cn[sub_graph_id]);
           __syncthreads();
         */

        while (search_depth[sub_graph_id] <= t_vertex_num[sub_graph_id]) 
        {
#ifndef BENCHMARK
            printf("search depth is %d\n", search_depth[sub_graph_id]);
            // std::cout<<"deleted_cols "<<std::endl;
            // cudaDeviceSynchronize();
            printf("deleted_cols\n");
            print_vec(t_deleted_cols[sub_graph_id], t_cn[sub_graph_id]);
            // cudaDeviceSynchronize();
            // cudaDeviceSynchronize();
            printf("deleted_rows\n");
            print_vec(t_deleted_rows[sub_graph_id], t_rn[sub_graph_id]);
            // cudaDeviceSynchronize();
            // cudaDeviceSynchronize();
            printf("results\n");
            print_vec(t_results[sub_graph_id], t_rn[sub_graph_id]);
            // cudaDeviceSynchronize();
#endif

            if (threadIdx.x == 0)
            {
                t_existance_of_candidate_rows[sub_graph_id] = 0;
                t_selected_row_id[sub_graph_id] = t_rn[sub_graph_id];
                t_conflict_node_id[sub_graph_id] = 0;
                t_conflict_col_id[sub_graph_id] = 0;
                t_conflict_edge[sub_graph_id][0] = 0;
                t_conflict_edge[sub_graph_id][1] = 0;
            }
            __syncthreads();
            // existance_of_candidate_rows=0;
            // selected_row_id=-1;
            check_existance_of_candidate_rows<MCSolverTraitsType>(
                    t_deleted_rows[sub_graph_id], t_row_group[sub_graph_id], search_depth[sub_graph_id],
                    &t_existance_of_candidate_rows[sub_graph_id], &t_selected_row_id[sub_graph_id], t_rn[sub_graph_id]);
            __syncthreads();
            // printf()
            // cudaMemcpy(existance_of_candidate_rows,
            // existance_of_candidate_rows_gpu, sizeof(int), cudaMemcpyDeviceToHost);
            // std::cout<<"check_existance_of_candidate_rows "<<std::endl;
            if (t_existance_of_candidate_rows[sub_graph_id] == 1) // check if there are candidate
            { 
                // rows existing, if no, do
                // backtrace
                // select_row <<<block_count, thread_count >>> (deleted_rows, row_group,
                // search_depth[sub_graph_id], total_dl_matrix_row_num, selected_row_id_gpu); //select
                // row and add to results
                // cudaMemcpy(selected_row_id, selected_row_id_gpu, sizeof(int),
                // cudaMemcpyDeviceToHost);
#ifndef BENCHMARK
                printf("selected row id is %d \n", t_selected_row_id[sub_graph_id]);
#endif
                //__syncthreads();
                // cudaMemset(&results[*selected_row_id],search_depth[sub_graph_id],sizeof(int));
                // set_vector_value<<<1,1>>>(results, *selected_row_id, search_depth[sub_graph_id]);
                delete_rows_and_columns<MCSolverTraitsType>(t_dl_matrix[sub_graph_id], t_next_row[sub_graph_id], t_next_col[sub_graph_id],
                        t_deleted_rows[sub_graph_id], t_deleted_cols[sub_graph_id], search_depth[sub_graph_id],
                        t_selected_row_id[sub_graph_id], t_rn[sub_graph_id],
                        t_cn[sub_graph_id]); // delete covered rows and columns
                //if (threadIdx.x == 0)
                //{
                //    delete_rows_and_columns_count += 1; 
                //}
                __syncthreads();

                if (threadIdx.x == 0)
                {
                    t_results[sub_graph_id][t_selected_row_id[sub_graph_id]] = search_depth[sub_graph_id];
                    // deleted_rows[*selected_row_id] = -search_depth[sub_graph_id];
                    t_deleted_rows[sub_graph_id][t_selected_row_id[sub_graph_id]] = -search_depth[sub_graph_id];
                    // set_vector_value<<<1,1>>>(deleted_rows, *selected_row_id,
                    // -search_depth[sub_graph_id]);
                    search_depth[sub_graph_id]++; // next step
                }
                // print_vec(deleted_cols, total_dl_matrix_col_num);
                // print_vec(deleted_rows, total_dl_matrix_row_num);
                // print_vec(conflict_count, total_dl_matrix_col_num);
                // print_vec(results, total_dl_matrix_row_num);
            } 
            else // do backtrace
            {
                if (threadIdx.x == 0)
                {
                    search_depth[sub_graph_id]--;
                }
                __syncthreads();

                if (search_depth[sub_graph_id] > 0) 
                {

                    get_conflict_node_id<MCSolverTraitsType>(t_deleted_rows[sub_graph_id], t_row_group[sub_graph_id], search_depth[sub_graph_id],
                            &t_conflict_node_id[sub_graph_id], t_rn[sub_graph_id]);
                    __syncthreads();
                    if (t_conflict_node_id[sub_graph_id] > 0) 
                    {

                        get_conflict_edge<MCSolverTraitsType>(t_dl_matrix[sub_graph_id], t_deleted_rows[sub_graph_id], t_row_group[sub_graph_id],
                                t_conflict_node_id[sub_graph_id], search_depth[sub_graph_id],
                                t_conflict_edge[sub_graph_id], t_vertex_num[sub_graph_id], t_rn[sub_graph_id], t_cn[sub_graph_id]);
                        __syncthreads();
                        get_conflict_col_id<MCSolverTraitsType>(t_dl_matrix[sub_graph_id], t_deleted_cols[sub_graph_id],
                                &t_conflict_col_id[sub_graph_id], t_conflict_edge[sub_graph_id], t_cn[sub_graph_id],
                                t_vertex_num[sub_graph_id]);
                        __syncthreads();

                        // conflict_count[*conflict_col_id]++;
                        // update conflict edge count
                        t_conflict_count[sub_graph_id][t_conflict_col_id[sub_graph_id]]++;
                        // add_gpu<<<1,1>>>(&deleted_rows[*selected_row_id],1);
                        recover_deleted_rows(t_deleted_rows[sub_graph_id], search_depth[sub_graph_id],
                                t_rn[sub_graph_id]); // recover deleted
                        // rows  previously
                        // selected rows

                        recover_deleted_cols(t_deleted_cols[sub_graph_id], search_depth[sub_graph_id],
                                t_cn[sub_graph_id]); // recover deleted
                        // cols except
                        // afftected by
                        // previously
                        // selected rows

                        recover_results(t_results[sub_graph_id], search_depth[sub_graph_id],
                                t_rn[sub_graph_id]); // recover results
                        __syncthreads();
                        // cudaMemcpy(&current_conflict_count[sub_graph_id],
                        // &conflict_count[*conflict_col_id], sizeof(int),
                        // cudaMemcpyDeviceToHost);
                        if (t_conflict_count[sub_graph_id][t_conflict_col_id[sub_graph_id]] >
                                hard_conflict_threshold) 
                        {
                            init_vectors(t_conflict_count[sub_graph_id], t_cn[sub_graph_id]);
                            init_vectors_reserved(t_deleted_cols[sub_graph_id], t_cn[sub_graph_id]);
                            init_vectors(t_deleted_rows[sub_graph_id], t_rn[sub_graph_id]);
                            init_vectors(t_results[sub_graph_id], t_rn[sub_graph_id]);
                            __syncthreads();
                            remove_cols<MCSolverTraitsType>(t_deleted_cols[sub_graph_id], t_col_group[sub_graph_id], t_conflict_col_id[sub_graph_id],
                                    t_cn[sub_graph_id]);
                            __syncthreads();

                            // /cudaMemset(&deleted_cols[*conflict_col_id],-1,sizeof(int));
                            if (threadIdx.x == 0)
                            {
                                t_deleted_cols[sub_graph_id][t_conflict_col_id[sub_graph_id]] = -1;
                                search_depth[sub_graph_id] = 1;
                            }
                        }
                    } 
                    else 
                    {
                        recover_deleted_rows(t_deleted_rows[sub_graph_id], search_depth[sub_graph_id],
                                t_rn[sub_graph_id]); // recover deleted
                        // rows  previously
                        // selected rows

                        recover_deleted_cols(t_deleted_cols[sub_graph_id], search_depth[sub_graph_id],
                                t_cn[sub_graph_id]); // recover deleted
                        // cols except
                        // afftected by
                        // previously
                        // selected rows

                        recover_results(t_results[sub_graph_id], search_depth[sub_graph_id],
                                t_rn[sub_graph_id]); // recover results
                    }
                } 
                else // if all vertices are gone through, directly remove the edge
                { 
                    // with largest conflict count.
                    if (threadIdx.x == 0)
                    {
                        search_depth[sub_graph_id] = 1;
                    }
                    t_max[sub_graph_id]=0;
                    get_largest_value(t_conflict_count[sub_graph_id], t_cn[sub_graph_id], &t_max[sub_graph_id]);

                    // cudaMemcpy(conflict_col_id, conflict_col_id_gpu, sizeof(int),
                    // cudaMemcpyDeviceToHost);
                    __syncthreads();
                    find_index(t_conflict_count[sub_graph_id], t_cn[sub_graph_id], &t_max[sub_graph_id], &t_conflict_col_id[sub_graph_id]);
                    init_vectors(t_conflict_count[sub_graph_id], t_cn[sub_graph_id]);
                    init_vectors_reserved(t_deleted_cols[sub_graph_id], t_cn[sub_graph_id]);
                    init_vectors(t_deleted_rows[sub_graph_id], t_rn[sub_graph_id]);
                    init_vectors(t_results[sub_graph_id], t_rn[sub_graph_id]);
                    __syncthreads();
                    remove_cols<MCSolverTraitsType>(t_deleted_cols[sub_graph_id], t_col_group[sub_graph_id], t_conflict_col_id[sub_graph_id], t_cn[sub_graph_id]);
                }
                // print_vec(deleted_cols, total_dl_matrix_col_num);
                // print_vec(deleted_rows, total_dl_matrix_row_num);
                // print_vec(conflict_count, total_dl_matrix_col_num);
                // print_vec(results, total_dl_matrix_row_num);
            }
            __syncthreads();
        }
        for(int i=threadIdx.x; i < t_rn[sub_graph_id]; i+=blockDim.x)
        {
            t_final_results[sub_graph_id][i] = t_results[sub_graph_id][i];
        }
        //if (threadIdx.x == 0)
        //{
        //    printf("[%d, %d] = %d, %dx%d\n", k, sub_graph_id, delete_rows_and_columns_count, t_rn[sub_graph_id], t_cn[sub_graph_id]);
        //}
    }
}

} // namespace gpu_mg
