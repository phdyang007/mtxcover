#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <iostream>

//#include "cub/cub/cub.cuh"
#include "../gpu-mg/BitMatrix.cuh"

namespace gpu_mg {

//constexpr int size_bit = 1 << 31;

#define GRAPH_PER_BLOCK 1

template <typename MCSolverTraitsType>
struct MCSolverState
{
    typedef BitMatrices<unsigned int, MCSolverTraitsType::max_num_rows, MCSolverTraitsType::max_num_cols> matrix_type; 

    short deleted_rows[MCSolverTraitsType::max_num_rows];
    short deleted_cols[MCSolverTraitsType::max_num_cols];
    typename matrix_type::row_type col_markers; ///< if deleted_cols[i] == 0, col_markers[i] == 1; otherwise, zero
    typename matrix_type::col_type row_markers; ///< if deleted_rows[i] == 0, row_markers[i] == 1; otherwise, zero
    typename matrix_type::row_type col_markers_copy; ///< for temporary usage 
    typename matrix_type::col_type row_markers_copy; ///< for temporary usage 
    short conflict_count[MCSolverTraitsType::max_num_cols];
    short results[MCSolverTraitsType::max_num_rows];
    int conflict_edge[2];
    short search_depth;
    int max;
    int existance_of_candidate_rows;
    int conflict_node_id;
    int conflict_col_id;
    int vertex_num;
    int selected_row_id;

    int cn;
    int rn;
    typename MCSolverTraitsType::ResultType *final_results;
    typename MCSolverTraitsType::RowGroupType *row_group;
    typename MCSolverTraitsType::ColGroupType *col_group;
    typename matrix_type::row_type *dl_matrix_rows;
    typename matrix_type::col_type *dl_matrix_cols;
    //typename MCSolverTraitsType::NextColType *next_col;
    //typename MCSolverTraitsType::NextRowType *next_row;
};

template <typename MCSolverTraitsType>
__device__ void delete_rows_and_columns(MCSolverState<MCSolverTraitsType>& state) 
{
    auto const& selected_row = state.dl_matrix_rows[state.selected_row_id];
    state.col_markers_copy.block_copy(state.col_markers);
    state.col_markers_copy.block_and(selected_row); 
    state.row_markers_copy.block_reset();
    __syncthreads();
    for (int i = threadIdx.x; i < state.cn; i += blockDim.x)
    {
        if (state.col_markers_copy.at(i)) 
        {
            state.deleted_cols[i] = state.search_depth; 
            state.col_markers.atomic_reset(i);

            state.row_markers_copy.atomic_or(state.dl_matrix_cols[i]);
        }
    }
    __syncthreads();
    state.row_markers_copy.block_and(state.row_markers);
    __syncthreads();
    for (int i = threadIdx.x; i < state.rn; i += blockDim.x)
    {
        if (state.row_markers_copy.at(i))
        {
            state.deleted_rows[i] = state.search_depth;
            state.row_markers.atomic_reset(i);
        }
    }

    //for (int i = threadIdx.x; i < total_dl_matrix_col_num;
    //        // // The below line will have negative effect of the col number is small
    //        //  i += (next_col[selected_row_idx + i] + blockDim.x - 1) / blockDim.x
    //        i += blockDim.x) 
    //{
    //    if (state.deleted_cols[i] == 0 && state.selected_row[i] == 1) 
    //    {
    //        state.deleted_cols[i] = state.search_depth;
    //        //atomicInc(&tmp_deleted_cols_count)
    //        for (int j = 0; j < state.total_dl_matrix_row_num;
    //                j += state.next_row[i * state.total_dl_matrix_row_num + j]) 
    //        {
    //            if (state.deleted_rows[j] == 0 &&
    //                    state.dl_matrix[j * state.total_dl_matrix_col_num + i] == 1) 
    //            {
    //                state.deleted_rows[j] = state.search_depth;

    //            }
    //        }
    //    }    
    //}
}

template <typename T>
__device__ void init_vectors(T *vec, const int vec_length) 
{
    for (int i = threadIdx.x; i < vec_length; i += blockDim.x) 
    {
        vec[i] = 0;
    }
}

template <typename T, typename V>
__device__ void get_largest_value(T *vec, const int vec_length, V *max) 
{
    for (int i = threadIdx.x; i < vec_length; i += blockDim.x) 
    {
        atomicMax(max, vec[i]);
    }
}

template <typename T, typename V>
__device__ void find_index(T *vec, const int vec_length, V *value, V *index) 
{
    for (int i = threadIdx.x; i < vec_length; i += blockDim.x) 
    {
        if (vec[i] == *value) 
        {
            atomicMax(index, i);
        }
    }
}

template <typename MCSolverTraitsType>
__device__ void init_deleted_cols_reserved(MCSolverState<MCSolverTraitsType>& state) 
{
    for (int i = threadIdx.x; i < state.cn; i += blockDim.x) 
    {
        if (state.deleted_cols[i] != -1) 
        {
            state.deleted_cols[i] = 0;
            state.col_markers.atomic_set(i);
        }
    }
}

template <typename MCSolverTraitsType>
__device__ void init_deleted_cols(MCSolverState<MCSolverTraitsType>& state) 
{
    for (int i = threadIdx.x; i < state.cn; i += blockDim.x) 
    {
        state.deleted_cols[i] = 0;
    }
    state.col_markers.block_set();
}

template <typename MCSolverTraitsType>
__device__ void init_deleted_rows(MCSolverState<MCSolverTraitsType>& state) 
{
    for (int i = threadIdx.x; i < state.rn; i += blockDim.x) 
    {
        state.deleted_rows[i] = 0;
    }
    state.row_markers.block_set();
}

template <typename MCSolverTraitsType>
__device__ void check_existance_of_candidate_rows(MCSolverState<MCSolverTraitsType>& state) 
{
    for (int i = threadIdx.x; i < state.rn; i += blockDim.x) 
    {
        // std::cout<<deleted_rows[i]<<' '<<row_group[i]<<std::endl;
        if (state.deleted_rows[i] == 0 && state.row_group[i] == state.search_depth) 
        {
            // std::cout<<"Candidate Row Found...."<<std::endl;
            atomicExch(&state.existance_of_candidate_rows, 1);
            atomicMin(&state.selected_row_id, i);
            // If find a number can break;
            //break;
        }
    }
}

template <typename MCSolverTraitsType>
__device__ void get_vertex_row_group(typename MCSolverTraitsType::RowGroupType *row_group, 
        const typename BitMatrices<unsigned int, MCSolverTraitsType::max_num_rows, MCSolverTraitsType::max_num_cols>::row_type *dl_matrix_rows,
        const int vertex_num,
        const int total_dl_matrix_row_num,
        const int total_dl_matrix_col_num) 
{
  // printf("%d %d\n", vertex_num, total_dl_matrix_row_num);
    for (int i = threadIdx.x; i < total_dl_matrix_row_num; i += blockDim.x) 
    {
        for (int j = 0; j < vertex_num; j++) 
        {
            row_group[i] += dl_matrix_rows[i].at(j) * (j + 1);
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

template <typename MCSolverTraitsType>
__device__ void recover_deleted_rows(MCSolverState<MCSolverTraitsType>& state) 
{
    for (int i = threadIdx.x; i < state.rn; i += blockDim.x) 
    {
        if (abs(state.deleted_rows[i]) > state.search_depth ||
                state.deleted_rows[i] == state.search_depth) 
        {
            state.deleted_rows[i] = 0;
            state.row_markers.atomic_set(i);
        }
    }
}

template <typename MCSolverTraitsType>
__device__ void recover_deleted_cols(MCSolverState<MCSolverTraitsType>& state) 
{
    for (int i = threadIdx.x; i < state.cn; i += blockDim.x) 
    {
        if (state.deleted_cols[i] >= state.search_depth) 
        {
            state.deleted_cols[i] = 0;
            state.col_markers.atomic_set(i);
        }
    }
}

template <typename MCSolverTraitsType>
__device__ void recover_results(MCSolverState<MCSolverTraitsType>& state) 
{
    for (int i = threadIdx.x; i < state.rn; i += blockDim.x) 
    {
        if (state.results[i] == state.search_depth) 
        {
            state.results[i] = 0;
        }
    }
}

// problem: need to optimized to map on GPU array
template <typename MCSolverTraitsType>
__device__ void get_conflict_node_id(MCSolverState<MCSolverTraitsType>& state) 
{
    auto search_depth_p1 = state.search_depth + 1; 
    for (int i = threadIdx.x; i < state.rn; i += blockDim.x) 
    {
        if (state.row_group[i] == search_depth_p1 &&
                state.deleted_rows[i] < search_depth_p1) 
        {
            atomicMax(&state.conflict_node_id, state.deleted_rows[i]);
        }
    }
}

template <typename MCSolverTraitsType>
__device__ void get_conflict_edge(MCSolverState<MCSolverTraitsType>& state) 
{
    for (int i = threadIdx.x; i < state.rn; i += blockDim.x) 
    {
        // find the conflict edge that connects current node and the most closest
        // node.
        if (state.deleted_rows[i] == -state.conflict_node_id) 
        {
            atomicMax(state.conflict_edge, i);
        }
        if (state.row_group[i] == state.search_depth + 1 &&
                state.deleted_rows[i] == state.conflict_node_id) 
        {
            atomicMax(state.conflict_edge + 1, i);
        }
    }

}

template <typename MCSolverTraitsType>
__device__ void get_conflict_col_id(MCSolverState<MCSolverTraitsType>& state) 
{
    // if(threadIdx.x==0){
    //  printf("conflict edge a %d edge b
    //  %d\n",conflict_edge[0],conflict_edge[1]);
    // }
    auto const& edge_a_dlmatrix = state.dl_matrix_rows[state.conflict_edge[0]];
    auto const& edge_b_dlmatrix = state.dl_matrix_rows[state.conflict_edge[1]]; 
    for (int j = threadIdx.x; j < state.cn; j += blockDim.x) 
    {
        if ((edge_a_dlmatrix.at(j) & edge_b_dlmatrix.at(j)) &&
                state.deleted_cols[j] > 0) 
        {
            atomicMax(&state.conflict_col_id, j);
        }
    }

}

template <typename MCSolverTraitsType>
__device__ void remove_cols(MCSolverState<MCSolverTraitsType>& state) 
{
    for (int i = threadIdx.x; i < state.cn; i += blockDim.x) 
    {
        if (state.col_group[i] == state.col_group[state.conflict_col_id]) 
        {
            state.deleted_cols[i] = -1;
            state.col_markers.atomic_reset(i);
        }
    }
}

template <typename T>
__device__ void print_vec(T *vec, int vec_length) 
{
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < vec_length; i++) 
        {
            printf("%d ", (int)vec[i]);
        }
        printf("\n");
    }
}

template <typename T>
__device__ void print_mat(T *mat[], int total_dl_matrix_row_num,
                          int total_dl_matrix_col_num) 
{
    for (int i = 0; i < total_dl_matrix_row_num; i++) 
    {
        for (int j = 0; j < total_dl_matrix_col_num; j++) 
        {
            printf("%d ", (int)mat[i][j]);
        }
        printf("\n");
    }
}

__device__ void add_gpu(int *device_var, int val) 
{
    atomicAdd(device_var, val);
}

__device__ void set_vector_value(int *device_var, int idx, int val) 
{
    device_var[idx] = val;
}

template <typename MCSolverTraitsType>
__global__ void 
init_vertex_group(typename MCSolverTraitsType::RowGroupType *row_group, 
        BitMatrices<unsigned int, MCSolverTraitsType::max_num_rows, MCSolverTraitsType::max_num_cols> dl_matrix, 
        int* vertex_num, int* t_cn, int* t_rn, int *offset_row, int *offset_matrix, int graph_count) 
{
    int k = blockIdx.x;
    if(k < graph_count)
    {
        get_vertex_row_group<MCSolverTraitsType>(row_group+offset_row[k], dl_matrix.rows(k), vertex_num[k], t_rn[k], t_cn[k]);
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

template <typename MCSolverTraitsType>
__global__ void
mc_solver(BitMatrices<unsigned int, MCSolverTraitsType::max_num_rows, MCSolverTraitsType::max_num_cols> dl_matrix, 
        typename MCSolverTraitsType::NextColType *next_col, typename MCSolverTraitsType::NextRowType *next_row, typename MCSolverTraitsType::ResultType *results,
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

    __shared__ MCSolverState<MCSolverTraitsType> states[GRAPH_PER_BLOCK]; 
    __shared__ int delete_rows_and_columns_count; 

    //end add shared mem
    int sub_graph_id = threadIdx.y;
    int k = blockIdx.x * graph_per_block + sub_graph_id;

    auto& state = states[sub_graph_id]; 

    //int k_p = threadIdx.y;
    if (k < graph_count && (k == 1784 || k == 1778 || k == 1792 || k == 2352 || k == 2350 || k == 2351 || k == 2349))
    {
        if (threadIdx.x == 0)
        {
            state.cn = total_dl_matrix_col_num[k];
            state.rn = total_dl_matrix_row_num[k];
            state.final_results  = results + offset_row[k];
            state.row_group      = row_group + offset_row[k];
            state.col_group = col_group + offset_col[k];
            state.dl_matrix_rows = dl_matrix.rows(k);
            state.dl_matrix_cols = dl_matrix.cols(k);
            //state.next_col = next_col + offset_matrix[k];
            //state.next_row= next_row + offset_matrix[k];

            state.vertex_num = vertex_num[k];

            state.search_depth = 1; 

            delete_rows_and_columns_count = 0; 
        }
        __syncthreads();
        //int *conflict_edge[sub_graph_id] = conflict_edge + 2 * k;


#ifndef BENCHMARK
        printf("blockID is %d\n", k);
        printf("vertexnum is %d\n", vertex_num[k]);
        printf("init conflict count \n");
#endif
        init_vectors(state.conflict_count, state.cn);
#ifndef BENCHMARK
        for (int i = 0; i < state.cn; i++) {
            printf("%d ", state.conflict_count[i]);
        }
        printf("\n");
#endif
        init_deleted_cols(state);
        init_deleted_rows(state);
        init_vectors(state.results, state.rn);
        __syncthreads();

        //get_vertex_row_group<MCSolverTraitsType>(state.row_group, state.dl_matrix, state.vertex_num, state.rn, state.cn);
        //__syncthreads();
        /*
           print_vec(deleted_cols+offset_col[k], state.cn);
           __syncthreads();
           print_vec(deleted_rows+offset_row[k], state.rn);
           __syncthreads();
           print_vec(results+offset_row[k], state.rn);
           __syncthreads();
           print_vec(row_group+offset_row[k], state.rn);
           __syncthreads();
           print_vec(col_group+offset_col[k], state.cn);
           __syncthreads();
         */

        while (state.search_depth <= state.vertex_num) 
        {
#ifndef BENCHMARK
            printf("search depth is %d\n", state.search_depth);
            // std::cout<<"deleted_cols "<<std::endl;
            // cudaDeviceSynchronize();
            printf("deleted_cols\n");
            print_vec(state.deleted_cols, state.cn);
            // cudaDeviceSynchronize();
            // cudaDeviceSynchronize();
            printf("deleted_rows\n");
            print_vec(state.deleted_rows, state.rn);
            // cudaDeviceSynchronize();
            // cudaDeviceSynchronize();
            printf("results\n");
            print_vec(state.results, state.rn);
            // cudaDeviceSynchronize();
#endif

            if (threadIdx.x == 0)
            {
                state.existance_of_candidate_rows = 0;
                state.selected_row_id = state.rn;
                state.conflict_node_id = 0;
                state.conflict_col_id = 0;
                state.conflict_edge[0] = 0;
                state.conflict_edge[1] = 0;
            }
            __syncthreads();
            // existance_of_candidate_rows=0;
            // selected_row_id=-1;
            check_existance_of_candidate_rows<MCSolverTraitsType>(state);
            __syncthreads();
            // printf()
            // cudaMemcpy(existance_of_candidate_rows,
            // existance_of_candidate_rows_gpu, sizeof(int), cudaMemcpyDeviceToHost);
            if (state.existance_of_candidate_rows == 1) // check if there are candidate
            { 
                // rows existing, if no, do
                // backtrace
                // select_row <<<block_count, thread_count >>> (state.deleted_rows, state.row_group,
                // state.search_depth, state.total_dl_matrix_row_num, state.selected_row_id_gpu); //select
                // row and add to results
                // cudaMemcpy(selected_row_id, selected_row_id_gpu, sizeof(int),
                // cudaMemcpyDeviceToHost);
#ifndef BENCHMARK
                printf("selected row id is %d \n", state.selected_row_id);
#endif
                //__syncthreads();
                delete_rows_and_columns<MCSolverTraitsType>(state); // delete covered rows and columns
                if (threadIdx.x == 0)
                {
                    delete_rows_and_columns_count += 1; 
                }
                __syncthreads();

                if (threadIdx.x == 0)
                {
                    state.results[state.selected_row_id] = state.search_depth;
                    // deleted_rows[*selected_row_id] = -state.search_depth;
                    state.deleted_rows[state.selected_row_id] = -state.search_depth;
                    state.row_markers.reset(state.selected_row_id);
                    state.search_depth++; // next step
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
                    state.search_depth--;
                }
                __syncthreads();

                if (state.search_depth > 0) 
                {

                    get_conflict_node_id<MCSolverTraitsType>(state);
                    __syncthreads();
                    if (state.conflict_node_id > 0) 
                    {

                        get_conflict_edge<MCSolverTraitsType>(state);
                        __syncthreads();
                        get_conflict_col_id<MCSolverTraitsType>(state);
                        __syncthreads();

                        // conflict_count[*conflict_col_id]++;
                        // update conflict edge count
                        if (threadIdx.x == 0)
                        {
                            state.conflict_count[state.conflict_col_id]++;
                        }
                        recover_deleted_rows(state); // recover deleted
                        // rows  previously
                        // selected rows

                        recover_deleted_cols(state); // recover deleted
                        // cols except
                        // afftected by
                        // previously
                        // selected rows

                        recover_results(state); // recover results
                        __syncthreads();
                        if (state.conflict_count[state.conflict_col_id] >
                                hard_conflict_threshold) 
                        {
                            init_vectors(state.conflict_count, state.cn);
                            init_deleted_cols_reserved(state);
                            init_deleted_rows(state);
                            init_vectors(state.results, state.rn);
                            __syncthreads();
                            remove_cols<MCSolverTraitsType>(state);
                            __syncthreads();

                            // /cudaMemset(&deleted_cols[*conflict_col_id],-1,sizeof(int));
                            if (threadIdx.x == 0)
                            {
                                state.deleted_cols[state.conflict_col_id] = -1;
                                state.col_markers.reset(state.conflict_col_id);
                                state.search_depth = 1;
                            }
                        }
                    } 
                    else 
                    {
                        recover_deleted_rows(state); // recover deleted
                        // rows  previously
                        // selected rows

                        recover_deleted_cols(state); // recover deleted
                        // cols except
                        // afftected by
                        // previously
                        // selected rows

                        recover_results(state); // recover results
                    }
                } 
                else // if all vertices are gone through, directly remove the edge
                { 
                    // with largest conflict count.
                    if (threadIdx.x == 0)
                    {
                        state.search_depth = 1;
                        state.max = 0;
                    }
                    __syncthreads();
                    get_largest_value(state.conflict_count, state.cn, &state.max);

                    // cudaMemcpy(conflict_col_id, conflict_col_id_gpu, sizeof(int),
                    // cudaMemcpyDeviceToHost);
                    __syncthreads();
                    find_index(state.conflict_count, state.cn, &state.max, &state.conflict_col_id);
                    init_vectors(state.conflict_count, state.cn);
                    init_deleted_cols_reserved(state);
                    init_deleted_rows(state);
                    init_vectors(state.results, state.rn);
                    __syncthreads();
                    remove_cols<MCSolverTraitsType>(state);
                }
                // print_vec(deleted_cols, total_dl_matrix_col_num);
                // print_vec(deleted_rows, total_dl_matrix_row_num);
                // print_vec(conflict_count, total_dl_matrix_col_num);
                // print_vec(results, total_dl_matrix_row_num);
            }
            __syncthreads();
        }
        for(int i=threadIdx.x; i < state.rn; i+=blockDim.x)
        {
            state.final_results[i] = state.results[i];
        }
        if (threadIdx.x == 0)
        {
            printf("[%d, %d] = %d, %dx%d\n", k, sub_graph_id, delete_rows_and_columns_count, state.rn, state.cn);
        }
    }
}

} // namespace gpu_mg
