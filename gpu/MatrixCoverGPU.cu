
#include "MatrixCoverGPU.cuh"

__global__ void delete_rows_and_columns(int*dl_matrix, int* deleted_rows, int* deleted_cols, const int search_depth, const int selected_row_id, const int total_dl_matrix_row_num, const int total_dl_matrix_col_num)
{
	for (int i = threadIdx.x; i < total_dl_matrix_col_num; i=i+blockDim.x)
	{
		if (dl_matrix[selected_row_id*total_dl_matrix_col_num+i] == 1 && deleted_cols[i] == 0)
		{ 
			deleted_cols[i] = search_depth;
			for (int j = 0; j < total_dl_matrix_row_num; j++)
			{
				if (dl_matrix[j*total_dl_matrix_col_num+i] == 1 && deleted_rows[j] == 0)
				{
					atomicExch(deleted_rows+j, search_depth);
				}
			}
		}
	}
}


__global__ void init_vectors(int* vec, const int vec_length)
{
	for (int i = threadIdx.x; i < vec_length; i = i + blockDim.x)
	{
		vec[i] = 0;
	}
}


/*
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
*/

__global__ void get_largest_value(int* vec, int *conflict_col_id, const int vec_length, int max)
{
	
	for (int i = threadIdx.x; i< vec_length; i = i + blockDim.x)
	{
		atomicMax(&max, vec[i]);
	}
	for (int i = threadIdx.x; i< vec_length; i = i + blockDim.x)
	{
		if (vec[i]==max)
		{
			*conflict_col_id = i;
		}
	}
}


__global__ void init_vectors_reserved(int *vec, const int vec_length)
{
    for (int i = threadIdx.x; i < vec_length; i= i+ blockDim.x)
    {
        if (vec[i] != -1)
        {
            vec[i] = 0;
        }
    }
}

__global__ void check_existance_of_candidate_rows(int* deleted_rows, int* row_group, const int search_depth, int *token, int * selected_row_id, const int total_dl_matrix_row_num)
{
	for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x)
	{
		//std::cout<<deleted_rows[i]<<' '<<row_group[i]<<std::endl;
		if (deleted_rows[i] == 0 && row_group[i] == search_depth)
		{
			//std::cout<<"Candidate Row Found...."<<std::endl;
			atomicExch(token, 1);
			atomicMin(selected_row_id, i);
		}
	}
	__syncthreads();
}


__global__ void get_vertex_row_group(int* row_group, int* dl_matrix, const int vertex_num, const int total_dl_matrix_row_num, const int total_dl_matrix_col_num)
{
	//printf("%d %d\n", vertex_num, total_dl_matrix_row_num);
	for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x)
	{
		for (int j = 0; j < vertex_num; j++)
		{
			row_group[i]+= dl_matrix[i*total_dl_matrix_col_num+j] * (j + 1);
		}
	}
}

/*
__global__ void select_row(int* deleted_rows, int* row_group, const int search_depth, const int total_dl_matrix_row_num, int* selected_row_id)
{
	for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x)
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

__global__ void recover_deleted_rows(int* deleted_rows, const int search_depth, const int total_dl_matrix_row_num)
{
	for (int i = threadIdx.x; i < total_dl_matrix_row_num; i=i+blockDim.x)
	{
		if (abs(deleted_rows[i]) > search_depth || deleted_rows[i] == search_depth)
		{
			deleted_rows[i] = 0;
		}
	}
}

__global__ void recover_deleted_cols(int* deleted_cols, const int search_depth, const int total_dl_matrix_col_num)
{
	for (int i = threadIdx.x; i < total_dl_matrix_col_num; i=i+blockDim.x)
	{
		if (deleted_cols[i] >= search_depth)
		{
			deleted_cols[i] = 0;
		}
	}
}

__global__ void recover_results(int* results, const int search_depth, const int total_dl_matrix_row_num)
{
	for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x)
	{
		if (results[i] == search_depth)
		{
			results[i] = 0;
		}
	}
}



//problem: need to optimized to map on GPU array
__global__ void get_conflict_node_id(int* deleted_rows, int* row_group, const int search_depth, int *conflict_node_id, const int total_dl_matrix_row_num) 
{
	for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i+ blockDim.x) {
		if (row_group[i] == search_depth + 1 && deleted_rows[i] > *conflict_node_id) {
			atomicExch(conflict_node_id, deleted_rows[i]);
		}
	}
	__syncthreads();
}


//problem
__global__ void get_conflict_col(int* dl_matrix, int* deleted_rows, int* deleted_cols, int* row_group, const int conflict_node_id, const int search_depth, int *conflict_col_id, const int vertex_num, const int total_dl_matrix_row_num, const int total_dl_matrix_col_num) {
	//*conflict_col_id = 0;
	for (int i = threadIdx.x; i < total_dl_matrix_row_num; i=i+blockDim.x) {  
		//find the conflict edge that connects current node and the most closest node.
		if (row_group[i] == search_depth + 1 && deleted_rows[i] == conflict_node_id) {
			for (int j = total_dl_matrix_col_num - 1; j > vertex_num; j--) {
				if (dl_matrix[i*total_dl_matrix_col_num+j] * deleted_cols[j] == conflict_node_id) {
					atomicExch(conflict_col_id, j);
				}
			}
			//for (int j = 0; j < total_dl_matrix_col_num - vertex_num-1; j++) {
			//	if (dl_matrix[i*total_dl_matrix_col_num+total_dl_matrix_col_num -1 -j] * deleted_cols[total_dl_matrix_col_num - 1 - j] == conflict_node_id) {
			//		atomicExch(conflict_col_id, j);
			//	}
			//}
		}
	}
	__syncthreads();
}


__global__ void remove_cols(int* deleted_cols, int* col_group, const int conflict_col_id, const int total_dl_matrix_col_num)
{
	for (int i = threadIdx.x; i < total_dl_matrix_col_num; i = i + blockDim.x)
	{
		if (col_group[i] == col_group[conflict_col_id])
		{
			deleted_cols[i] = -1;
		}
	}
}


__global__ void print_vec(int *vec, int vec_length)
{
	for(int i=0; i<vec_length; i++)
	{
		printf("%d ", vec[i]);
	}
	printf("\n");
}


__global__ void print_mat(int *mat[], int total_dl_matrix_row_num, int total_dl_matrix_col_num)
{
	for(int i=0; i<total_dl_matrix_row_num; i++)
	{
		for(int j=0; j<total_dl_matrix_col_num; j++)
		{
			printf("%d ", mat[i][j]);
		}
		printf("\n");
	}

}

__global__ void add_gpu(int *device_var, int val)
{
	atomicAdd(device_var, val);
}

__global__ void set_vector_value(int* device_var, int idx, int val)
{
	device_var[idx]=val;
}



void mc_solver(int* dl_matrix,	int* results, int* deleted_cols, int* deleted_rows,	int* col_group,	int* row_group,	int* conflict_count, const int vertex_num_gpu, const int total_dl_matrix_row_num_gpu, const int total_dl_matrix_col_num_gpu)
{
	//to be refreshed if one conflict reaches many counts
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

	const int block_count = 1;
	const int thread_count = 32;
	//init lots of vectors
	init_vectors<<<block_count,thread_count>>>(conflict_count, total_dl_matrix_col_num);
	init_vectors<<<block_count,thread_count>>>(deleted_cols, total_dl_matrix_col_num);
	init_vectors<<<block_count,thread_count>>>(deleted_rows, total_dl_matrix_row_num);
	init_vectors<<<block_count,thread_count>>>(results, total_dl_matrix_row_num);
	//init_vectors<<<block_count,thread_count>>>(row_group, total_dl_matrix_row_num);
	//__syncthreads();
	//get_vertex_row_group<<<block_count,thread_count >>>(row_group, dl_matrix, vertex_num, total_dl_matrix_row_num);
	//__syncthreads();


	//print_mat<<<block_count,thread_count>>>(dl_matrix, total_dl_matrix_row_num, total_dl_matrix_col_num);


	print_vec<<<1,1>>>(deleted_cols, total_dl_matrix_col_num_gpu);
	cudaDeviceSynchronize();
	print_vec<<<1,1>>>(deleted_rows, total_dl_matrix_row_num_gpu);
	cudaDeviceSynchronize();
	print_vec<<<1,1>>>(results, total_dl_matrix_row_num_gpu);
	cudaDeviceSynchronize();
	print_vec<<<1,1>>>(row_group, total_dl_matrix_row_num_gpu);
	cudaDeviceSynchronize();
	print_vec<<<1,1>>>(col_group, total_dl_matrix_col_num_gpu);
	cudaDeviceSynchronize();


	for (search_depth = 1; search_depth <= vertex_num;)
	{

		std::cout<<"search depth is "<<search_depth<<std::endl;
		std::cout<<"deleted_cols "<<std::endl;
		print_vec<<<1,1>>>(deleted_cols, total_dl_matrix_col_num_gpu);
		cudaDeviceSynchronize();
		std::cout<<"deleted_rows "<<std::endl;
		print_vec<<<1,1>>>(deleted_rows, total_dl_matrix_row_num_gpu);
		cudaDeviceSynchronize();
		std::cout<<"results "<<std::endl;
		print_vec<<<1,1>>>(results, total_dl_matrix_row_num_gpu);
		cudaDeviceSynchronize();
		std::cin>>brk;
		cudaMemset(existance_of_candidate_rows_gpu,0,sizeof(int));
		cudaMemset(selected_row_id_gpu,total_dl_matrix_row_num,sizeof(int));
		//existance_of_candidate_rows=0;
		//selected_row_id=-1;
		check_existance_of_candidate_rows <<<block_count, thread_count >>> (deleted_rows, row_group, search_depth, existance_of_candidate_rows_gpu, selected_row_id_gpu, total_dl_matrix_row_num);
		//__syncthreads();
		cudaMemcpy(existance_of_candidate_rows, existance_of_candidate_rows_gpu, sizeof(int), cudaMemcpyDeviceToHost);
		std::cout<<"check_existance_of_candidate_rows "<<std::endl;
		if (*existance_of_candidate_rows==1)
		{                                                                                                 //check if there are candidate rows existing, if no, do backtrace
			//select_row <<<block_count, thread_count >>> (deleted_rows, row_group, search_depth, total_dl_matrix_row_num, selected_row_id_gpu); //select row and add to results
			cudaMemcpy(selected_row_id, selected_row_id_gpu, sizeof(int), cudaMemcpyDeviceToHost);
			std::cout<<"selected row id is "<<*selected_row_id<<std::endl;
			//__syncthreads();
			//cudaMemset(&results[*selected_row_id],search_depth,sizeof(int));
			set_vector_value<<<1,1>>>(results, *selected_row_id, search_depth);
			delete_rows_and_columns <<<block_count, thread_count >>> (dl_matrix, deleted_rows, deleted_cols, search_depth, *selected_row_id, total_dl_matrix_row_num, total_dl_matrix_col_num); //delete covered rows and columns
			//__syncthreads();
			//deleted_rows[*selected_row_id] = -search_depth;
			set_vector_value<<<1,1>>>(deleted_rows, *selected_row_id, -search_depth);

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
				get_conflict_node_id <<<block_count, thread_count >>> (deleted_rows, row_group, search_depth, conflict_node_id_gpu, total_dl_matrix_row_num);
				cudaMemcpy(conflict_node_id, conflict_node_id_gpu, sizeof(int), cudaMemcpyDeviceToHost);
				//__syncthreads();
				get_conflict_col <<<block_count, thread_count >>> (dl_matrix, deleted_rows, deleted_cols, row_group, *conflict_node_id, search_depth, conflict_col_id_gpu, vertex_num, total_dl_matrix_row_num, total_dl_matrix_col_num);
				cudaMemcpy(conflict_col_id, conflict_col_id_gpu, sizeof(int), cudaMemcpyDeviceToHost);

				//conflict_count[*conflict_col_id]++;                                                                   //update conflict edge count
				add_gpu<<<1,1>>>(&conflict_count[*conflict_col_id_gpu],1);
				recover_deleted_rows<<<block_count, thread_count >>>(deleted_rows, search_depth, total_dl_matrix_row_num);                           //recover deleted rows  previously selected rows
				//__syncthreads();
				recover_deleted_cols<<<block_count, thread_count >>>(deleted_cols, search_depth, total_dl_matrix_col_num);                           //recover deleted cols except afftected by previously selected rows
				//__syncthreads();
				recover_results<<<block_count, thread_count >>>(results, search_depth, total_dl_matrix_row_num);                                     //recover results
				//__syncthreads();
				cudaMemcpy(&current_conflict_count, &conflict_count[*conflict_col_id], sizeof(int), cudaMemcpyDeviceToHost);
				if (current_conflict_count > hard_conflict_threshold)
				{
					search_depth = 1;
					init_vectors<<<block_count, thread_count >>>(conflict_count, total_dl_matrix_col_num);
					init_vectors_reserved<<<block_count, thread_count >>>(deleted_cols, total_dl_matrix_col_num);
					init_vectors<<<block_count, thread_count >>>(deleted_rows, total_dl_matrix_row_num);
					init_vectors<<<block_count, thread_count >>>(results, total_dl_matrix_row_num);
					//__syncthreads();
					remove_cols<<<block_count, thread_count >>>(deleted_cols, col_group, *conflict_col_id, total_dl_matrix_col_num);
					//__syncthreads();
					//deleted_cols[*conflict_col_id] = -1;
					cudaMemset(&deleted_cols[*conflict_col_id],-1,sizeof(int));
				}
			}
			else
			{ //if all vertices are gone through, directly remove the edge with largest conflict count.
				search_depth = 1;
				get_largest_value<<<block_count, thread_count >>>(conflict_count, conflict_col_id_gpu, total_dl_matrix_col_num, 0);
				cudaMemcpy(conflict_col_id, conflict_col_id_gpu, sizeof(int), cudaMemcpyDeviceToHost);
				//__syncthreads();
				init_vectors<<<block_count, thread_count >>>(conflict_count, total_dl_matrix_col_num);
				init_vectors_reserved<<<block_count, thread_count >>>(deleted_cols, total_dl_matrix_col_num);
				init_vectors<<<block_count, thread_count >>>(deleted_rows, total_dl_matrix_row_num);
				init_vectors<<<block_count, thread_count >>>(results, total_dl_matrix_row_num);
				//__syncthreads();
				remove_cols<<<block_count, thread_count >>>(deleted_cols, col_group, *conflict_col_id, total_dl_matrix_col_num);
			}
			//print_vec(deleted_cols, total_dl_matrix_col_num);
			//print_vec(deleted_rows, total_dl_matrix_row_num);
			//print_vec(conflict_count, total_dl_matrix_col_num);
			//print_vec(results, total_dl_matrix_row_num);
		}
	}

	cudaFree(existance_of_candidate_rows_gpu);
	cudaFree(selected_row_id_gpu);
	cudaFree(conflict_col_id_gpu);
	cudaFree(conflict_node_id_gpu);
	delete existance_of_candidate_rows;
	delete conflict_col_id;
	delete selected_row_id;
	delete conflict_node_id;
}
