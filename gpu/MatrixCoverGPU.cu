#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cub/cub/cub.cuh"
#include "cub/cub/device/device_reduce.cuh"
#include "MatrixCoverGPU.h"

__global__ void delete_rows_and_columns(int** dl_matrix, int* deleted_rows, int* deleted_cols, int* search_depth, int* selected_row_id, const int* total_dl_matrix_row_num, const int* total_dl_matrix_col_num)
{
	for (int i = threadIdx.x; i < *total_dl_matrix_col_num; i=i+blockDim.x)
	{
		if (dl_matrix[*selected_row_id][i] == 1 && deleted_cols[i] == 0)//we only delete rows that are not deleted or removed
		{ 
			deleted_cols[i] = *search_depth;
			for (int j = threadIdx.y; j < *total_dl_matrix_row_num; j=j+blockDim.y)
			{
				if (dl_matrix[j][i] == 1 && deleted_rows[j] == 0)
				{
					deleted_rows[j] = *search_depth;
				}
			}
		}
	}
}


__global__ void init_vectors(int* vec, const int *vec_length)
{
	for (int i = threadIdx.x; i < *vec_length; i = i + blockDim.x)
	{
		vec[i] = 0;
	}
}



void get_largest_value_launcher(int* vec, cub::KeyValuePair<int, int> *argmax, int vec_length)
{
	void* d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, vec, argmax, vec_length);
	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run argmax-reduction
	cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, vec, argmax, vec_length);
	cudaFree(d_temp_storage);
}


__global__ void init_vectors_reserved(int *vec, const int *vec_length)
{
    for (int i = threadIdx.x; i < *vec_length; i= i+ blockDim.x)
    {
        if (vec[i] != -1)
        {
            vec[i] = 0;
        }
    }
}

__global__ void check_existance_of_candidate_rows(int* deleted_rows, int* row_group, int* search_depth, bool *token, const int* total_dl_matrix_row_num)
{
	*token = false;
	for (int i = threadIdx.x; i < *total_dl_matrix_row_num; i = i + blockDim.x)
	{
		//std::cout<<deleted_rows[i]<<' '<<row_group[i]<<std::endl;
		if (deleted_rows[i] == 0 && row_group[i] == *search_depth)
		{
			//std::cout<<"Candidate Row Found...."<<std::endl;
			*token = true;
		}
	}
}


__global__ void get_vertex_row_group(int* row_group, int** dl_matrix, const int *vertex_num, const int *total_dl_matrix_row_num)
{
	for (int i = threadIdx.x; i < *vertex_num; i = i + blockDim.x)
	{
		for (int j = threadIdx.y; j < *total_dl_matrix_row_num; j = j + blockDim.y)
		{
			row_group[j] += dl_matrix[j][i] * (i + 1);
		}
	}
}


__global__ void select_row(int* deleted_rows, int* row_group, int* search_depth, const int* total_dl_matrix_row_num, int* selected_row_id)
{
	for (int i = threadIdx.x; i < *total_dl_matrix_row_num; i = i + blockDim.x)
	{
		if (deleted_rows[i] == 0 && row_group[i] == *search_depth)
		{
			atomicExch(selected_row_id, i);
		}
	}
	__syncthreads();
}


__global__ void recover_deleted_rows(int* deleted_rows, int *search_depth, const int *total_dl_matrix_row_num)
{
	for (int i = threadIdx.x; i < *total_dl_matrix_row_num; i=i+blockDim.x)
	{
		if (abs(deleted_rows[i]) > *search_depth || deleted_rows[i] == *search_depth)
		{
			deleted_rows[i] = 0;
		}
	}
}

__global__ void recover_deleted_cols(int* deleted_cols, int *search_depth, const int *total_dl_matrix_col_num)
{
	for (int i = threadIdx.x; i < *total_dl_matrix_col_num; i=i+blockDim.x)
	{
		if (deleted_cols[i] >= *search_depth)
		{
			deleted_cols[i] = 0;
		}
	}
}

__global__ void recover_results(int* results, int *search_depth, const int *total_dl_matrix_row_num)
{
	for (int i = threadIdx.x; i < *total_dl_matrix_row_num; i = i + blockDim.x)
	{
		if (results[i] == *search_depth)
		{
			results[i] = 0;
		}
	}
}



//problem: need to optimized to map on GPU array
__global__ void get_conflict_node_id(int* deleted_rows, int* row_group, int *search_depth, int *conflict_node_id, const int *total_dl_matrix_row_num) {
	*conflict_node_id = 0;
	for (int i = threadIdx.x; i < *total_dl_matrix_row_num; i = i+ blockDim.x) {
		if (row_group[i] == *search_depth + 1 && deleted_rows[i] > *conflict_node_id) {
			atomicExch(conflict_node_id, deleted_rows[i]);
		}
	}
	__syncthreads();
}


//problem
__global__ void get_conflict_col(int** dl_matrix, int* deleted_rows, int* deleted_cols, int* row_group, int *conflict_node_id, int *search_depth, int *conflict_col_id, const int* vertex_num, const int *total_dl_matrix_row_num, const int *total_dl_matrix_col_num) {
	*conflict_col_id = 0;
	for (int i = threadIdx.x; i < *total_dl_matrix_row_num; i=i+blockDim.x) {  
		//find the conflict edge that connects current node and the most closest node.
		if (row_group[i] == search_depth + 1 && deleted_rows[i] == conflict_node_id) {
			//for (int j = total_dl_matrix_col_num - 1; j > vertex_num; j--) {
			//	if (dl_matrix[i][j] * deleted_cols[j] == conflict_node_id) {
			//		atomicExch(conflict_col_id, j);
			//	}
			//}
			for (int j = threadIdx.y; j < *total_dl_matrix_col_num - *vertex_num-1; j = j + blockDim.y) {
				if (dl_matrix[i][*total_dl_matrix_col_num -1 -j] * deleted_cols[*total_dl_matrix_col_num - 1 - j] == *conflict_node_id) {
					atomicExch(conflict_col_id, j);
				}
			}
		}
	}
	__syncthreads();
}


__global__ void remove_cols(int* deleted_cols, int* col_group, int *conflict_col_id, const int *total_dl_matrix_col_num)
{
	for (int i = threadIdx.x; i < *total_dl_matrix_col_num; i = i + blockDim.x)
	{
		if (col_group[i] == col_group[*conflict_col_id])
		{
			deleted_cols[i] = -1;
		}
	}
}






__global__ void mc_solver(
	int** dl_matrix,
	int* results,
	int* deleted_cols,
	int* col_group,
	const int vertex_num,
	const int total_dl_matrix_row_num,
	const int total_dl_matrix_col_num)
{
	//to be refreshed if one conflict reaches many counts
	__shared__ int search_depth = 0;
	__shared__ int selected_row_id = 0;
	__shared__ int conflict_count[total_dl_matrix_col_num];
	//int* conflict_count = new int[total_dl_matrix_col_num];

	//int* deleted_rows = new int[total_dl_matrix_row_num];
	__shared__ int deleted_rows[total_dl_matrix_row_num];

	//int* vertices_covered = new int[vertex_num];
	//__shared__ int vertices_covered[vertex_num];

	//int* row_group = new int[total_dl_matrix_row_num];
	__shared__ int row_group[total_dl_matrix_row_num];

	//int *col_group = new int[total_dl_matrix_col_num];
	//int selected_row_id_in_previous_search;
	__shared__ int conflict_node_id;
	__shared__ int conflict_col_id;
	__shared__ int hard_conflict_threshold = 500;
	__shared__ int existance_of_candidate_rows;

	const int block_count = 1;
	const int thread_count = 1024;
	//init lots of vectors
	init_vectors<<<block_count,thread_count>>>(conflict_count, &total_dl_matrix_col_num);
	init_vectors<<<block_count,thread_count>>>(deleted_cols, &total_dl_matrix_col_num);
	init_vectors<<<block_count,thread_count>>>(deleted_rows, &total_dl_matrix_row_num);
	init_vectors<<<block_count,thread_count>>>(results, &total_dl_matrix_row_num);
	init_vectors<<<block_count,thread_count>>>(row_group, &total_dl_matrix_row_num);
	get_vertex_row_group<<<block_count,thread_count >>>(row_group, dl_matrix, &vertex_num, &total_dl_matrix_row_num);



	//print_vec(row_group, total_dl_matrix_row_num);
	//print_vec(col_group, total_dl_matrix_col_num);
	for (search_depth = 1; search_depth <= vertex_num;)
	{
		//std::cout<<"search depth is "<<search_depth<<std::endl;


		check_existance_of_candidate_rows <<<block_count, thread_count >>> (deleted_rows, row_group, &search_depth, &existance_of_candidate_rows, &total_dl_matrix_row_num);
		if (existance_of_candidate_rows==true)
		{                                                                                                 //check if there are candidate rows existing, if no, do backtrace
			select_row <<<block_count, thread_count >>> (deleted_rows, row_group, &search_depth, &total_dl_matrix_row_num, &selected_row_id); //select row and add to results
			results[selected_row_id] = search_depth;
			delete_rows_and_columns <<<block_count, thread_count >>> (dl_matrix, deleted_rows, deleted_cols, &search_depth, &selected_row_id, &total_dl_matrix_row_num, &total_dl_matrix_col_num); //delete covered rows and columns
			deleted_rows[selected_row_id] = -search_depth;
			search_depth++; //next step
			//print_vec(deleted_cols, total_dl_matrix_col_num);
			//print_vec(deleted_rows, total_dl_matrix_row_num);
			//print_vec(conflict_count, total_dl_matrix_col_num);
			//print_vec(results, total_dl_matrix_row_num);
			continue;
		}
		else
		{ //do backtrace
			search_depth--;
			if (search_depth > 0)
			{
				//conflict_node_id = get_conflict_node_id(deleted_rows, row_group, search_depth, total_dl_matrix_row_num);
				get_conflict_node_id <<<block_count, thread_count >>> (deleted_rows, row_group, &search_depth, &conflict_node_id, &total_dl_matrix_row_num);
				get_conflict_col <<<block_count, thread_count >>> (dl_matrix, deleted_rows, deleted_cols, row_group, &conflict_node_id, &search_depth, &conflict_col_id, &vertex_num, &total_dl_matrix_row_num, &total_dl_matrix_col_num);
				//conflict_col_id = get_conflict_col(dl_matrix, deleted_rows, deleted_cols, row_group, conflict_node_id, search_depth, vertex_num, total_dl_matrix_row_num, total_dl_matrix_col_num); // get conflict edge
				//std::cout << "conflict col id is " << conflict_col_id << std::endl;
				conflict_count[conflict_col_id]++;                                                                   //update conflict edge count
				recover_deleted_rows(deleted_rows, search_depth, total_dl_matrix_row_num);                           //recover deleted rows  previously selected rows
				recover_deleted_cols(deleted_cols, search_depth, total_dl_matrix_col_num);                           //recover deleted cols except afftected by previously selected rows
				recover_results(results, search_depth, total_dl_matrix_row_num);                                     //recover results

				if (conflict_count[conflict_col_id] > hard_conflict_threshold)
				{
					search_depth = 1;
					init_vectors(conflict_count, total_dl_matrix_col_num);
					init_vectors_reserved(deleted_cols, total_dl_matrix_col_num);
					init_vectors(deleted_rows, total_dl_matrix_row_num);
					init_vectors(results, total_dl_matrix_row_num);
					remove_cols(deleted_cols, col_group, conflict_col_id, total_dl_matrix_col_num);
					deleted_cols[conflict_col_id] = -1;
				}
			}
			else
			{ //if all vertices are gone through, directly remove the edge with largest conflict count.
				search_depth = 1;
				conflict_col_id = get_largest_value(conflict_count, total_dl_matrix_col_num);
				init_vectors(conflict_count, total_dl_matrix_col_num);
				init_vectors_reserved(deleted_cols, total_dl_matrix_col_num);
				init_vectors(deleted_rows, total_dl_matrix_row_num);
				init_vectors(results, total_dl_matrix_row_num);
				remove_cols(deleted_cols, col_group, conflict_col_id, total_dl_matrix_col_num);
			}
			//print_vec(deleted_cols, total_dl_matrix_col_num);
			//print_vec(deleted_rows, total_dl_matrix_row_num);
			//print_vec(conflict_count, total_dl_matrix_col_num);
			//print_vec(results, total_dl_matrix_row_num);
		}
	}

	delete[] deleted_rows;
	delete[] row_group;
	delete[] vertices_covered;
	delete[] conflict_count;
}
