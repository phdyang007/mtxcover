#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <iostream>

//#include "cub/cub/cub.cuh"

namespace gpu_mg {

__device__ void delete_rows_and_columns(int *dl_matrix, int *deleted_rows,
                                        int *deleted_cols,
                                        const int search_depth,
                                        const int selected_row_id,
                                        const int total_dl_matrix_row_num,
                                        const int total_dl_matrix_col_num);

__device__ void init_vectors(int *vec, const int vec_length);

__device__ void get_largest_value(int *vec,
                                  const int vec_length, int* max = 0);

__device__ void find_index(int *vec, const int vec_length, int *value, int *index);

              

__device__ void init_vectors_reserved(int *vec, const int vec_length);

__device__ void check_existance_of_candidate_rows(
    int *deleted_rows, int *row_group, const int search_depth, int *token,
    int *selected_row_id, const int total_dl_matrix_row_num);

__device__ void get_vertex_row_group(int *row_group, int *dl_matrix,
                                     const int vertex_num,
                                     const int total_dl_matrix_row_num,
                                     const int total_dl_matrix_col_num);

// void print_vec(int *vec, int vec_length);

//__device__ void select_row(int* deleted_rows, int* row_group, const int
// search_depth, const int total_dl_matrix_row_num, int* selected_row_id);

__device__ void recover_deleted_rows(int *deleted_rows, const int search_depth,
                                     const int total_dl_matrix_row_num);

__device__ void recover_deleted_cols(int *deleted_cols, const int search_depth,
                                     const int total_dl_matrix_col_num);

__device__ void recover_results(int *results, const int search_depth,
                                const int total_dl_matrix_row_num);

__device__ void get_conflict_node_id(int *deleted_rows, int *row_group,
                                    const int search_depth,
                                    int *conflict_node_id,
                                    const int total_dl_matrix_row_num);

__device__ void get_conflict_edge(int *dl_matrix, int *deleted_rows,
                                int *row_group,
                                const int conflict_node_id,
                                const int search_depth, int *conflict_edge,
                                const int vertex_num,
                                const int total_dl_matrix_row_num,
                                const int total_dl_matrix_col_num);
                                
__device__ void get_conflict_col_id(int *dl_matrix, int *deleted_cols, int* conflict_col_id,
                                   int *conflict_edge, int total_dl_matrix_col_num, 
                                   int vertex_num);

__device__ void remove_cols(int *deleted_cols, int *col_group,
                            const int conflict_col_id,
                            const int total_dl_matrix_col_num);

__global__ void
mc_solver(int *dl_matrix, int *next_col, int *next_row, int *results,
          int *deleted_cols, int *deleted_rows, int *col_group, int *row_group,
          int *conflict_count, int *vertex_num, int *total_dl_matrix_row_num,
          int *total_dl_matrix_col_num, int *offset_col, int *offset_row,
          int *offset_matrix, int *search_depth, int *selected_row_id,
          int *current_conflict_count, int *conflict_node_id,
          int *conflict_col_id, int *existance_of_candidate_rows, int* conflict_edge,
          int *max,
          const int graph_count, const int hard_conflict_threshold);
// void mc_solver(int* dl_matrix, int* results, int* deleted_cols, int*
// deleted_rows, int* col_group,int* row_group, int* conflict_count,	const
// int vertex_num, const int total_dl_matrix_row_num, const int
// total_dl_matrix_col_num);

} // namespace gpu_mg
