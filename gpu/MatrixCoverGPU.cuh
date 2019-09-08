#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cub/cub/cub.cuh"


__global__ void delete_rows_and_columns(int*dl_matrix[], int* deleted_rows, int* deleted_cols, int* search_depth, int* selected_row_id, const int* total_dl_matrix_row_num, const int* total_dl_matrix_col_num);



__global__ void init_vectors(int* vec, const int *vec_length);



__global__ void get_largest_value(int* vec, int *conflict_col_id const int *vec_length);




__global__ void init_vectors_reserved(int *vec, const int *vec_length);





__global__ void check_existance_of_candidate_rows(int* deleted_rows, int* row_group, int* search_depth, bool *token, const int* total_dl_matrix_row_num);





__global__ void get_vertex_row_group(int* row_group, int* dl_matrix[], const int *vertex_num, const int *total_dl_matrix_row_num);







__global__ void select_row(int* deleted_rows, int* row_group, int* search_depth, const int* total_dl_matrix_row_num, int* selected_row_id);




__global__ void recover_deleted_rows(int* deleted_rows, int *search_depth, const int *total_dl_matrix_row_num);





__global__ void recover_deleted_cols(int* deleted_cols, int *search_depth, const int *total_dl_matrix_col_num);






__global__ void recover_results(int* results, int *search_depth, const int *total_dl_matrix_row_num);





__global__ void get_conflict_node_id(int* deleted_rows, int* row_group, int *search_depth, int *conflict_node_id, const int *total_dl_matrix_row_num);






__global__ void get_conflict_col(int* dl_matrix[], int* deleted_rows, int* deleted_cols, int* row_group, int *conflict_node_id, int *search_depth, int *conflict_col_id, const int* vertex_num, const int *total_dl_matrix_row_num, const int *total_dl_matrix_col_num);





__global__ void remove_cols(int* deleted_cols, int* col_group, int *conflict_col_id, const int *total_dl_matrix_col_num);





__global__ void mc_solver(int* dl_matrix[], int* results, int* deleted_cols, int* col_group,	const int vertex_num, const int total_dl_matrix_row_num, const int total_dl_matrix_col_num);










