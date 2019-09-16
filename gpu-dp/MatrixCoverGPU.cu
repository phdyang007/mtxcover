
#include "MatrixCoverGPU.cuh"

__global__ void delete_rows_and_columns(int*dl_matrix, int* deleted_rows, int* deleted_cols, const int *search_depth, const int *selected_row_id, const int *total_dl_matrix_row_num, const int *total_dl_matrix_col_num, int * offset_col, int * offset_row, int * offset_matrix, const int graph_count)
{
	for (int k = blockIdx.x; k< graph_count; k = k + gridDim.x)
	{
		for (int i = threadIdx.x; i < total_dl_matrix_col_num[k]; i=i+blockDim.x)
		{
			if (dl_matrix[offset_matrix[k]+selected_row_id[k]*total_dl_matrix_col_num[k]+i] == 1 && deleted_cols[offset_col[k]+i] == 0)//we only delete rows that are not deleted or removed
			{ 
				deleted_cols[offset_col[k]+i] = search_depth[k];
				for (int j = 0; j < total_dl_matrix_row_num[k]; j++)
				{
					if (dl_matrix[offset_matrix[k]+j*total_dl_matrix_col_num[k]+i] == 1 && deleted_rows[offset_row[k]+j] == 0)
					{
						atomicExch(offset_row[k]+deleted_rows+j, search_depth[k]);
					}
				}
			}
		}
	}
}


__global__ void init_vectors(int* vec, const int *vec_length, int* offset, const int graph_count)
{
	for (int k=blockIdx.x; k<graph_count; k= k+gridDim.x)
	{
		for (int i = threadIdx.x; i < vec_length[k]; i = i + blockDim.x)
		{
			vec[offset[k]+i] = 0;
		}
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

__global__ void get_largest_value(int* vec, int *conflict_col_id, const int *vec_length, int *max, int *offset, const int graph_count)
{
	for (int k=blockIdx.x; k<graph_count; k = k+gridDim.x)
	{
		for (int i = threadIdx.x; i< vec_length[k]; i = i + blockDim.x)
		{
			atomicMax(max+k, vec[offset[k]+i]);
		}
		for (int i = threadIdx.x; i< vec_length[k]; i = i + blockDim.x)
		{
			if (vec[offset[k]+i]==max)
			{
				conflict_col_id[k] = i;
			}
		}
	}

}




__global__ void init_vectors_reserved(int *vec, int *vec_length, int *offset, const int graph_count)
{
	for (int k = blockIdx.x, k<graph_count; k = k+gridDim.x)
	{
		for (int i = threadIdx.x; i < vec_length[k]; i= i+ blockDim.x)
		{
			if (vec[offset[k]+i] != -1)
			{
				vec[offset[k]+i] = 0;
			}
		}
	}
}




__global__ void check_existance_of_candidate_rows(int* deleted_rows, int* row_group, const int *search_depth, int *token, int *total_dl_matrix_row_num, int* selected_row_id, int* offset_row, const int graph_count)
{
	for (int k = blockIdx.x, k<graph_count; k = k+gridDim.x)
	{
		for (int i = threadIdx.x; i < total_dl_matrix_row_num[k]; i = i + blockDim.x)
		{
			//std::cout<<deleted_rows[i]<<' '<<row_group[i]<<std::endl;
			if (deleted_rows[offset[k]+i] == 0 && row_group[offset[k]+i] == search_depth[k])
			{
				//std::cout<<"Candidate Row Found...."<<std::endl;
				atomicExch(token+k, 1);
				atomicMin(selected_row_id+k, i);
			}
		}
	}
}


__global__ void get_vertex_row_group(int* row_group, int* dl_matrix, int* vertex_num, int* total_dl_matrix_row_num, int* total_dl_matrix_col_num, int * offset_row, int *offset_matrix, const int graph_count)
{
	for (int k = blockIdx.x; k<graph_count; k = k+ gridDim.x)
	{
		for (int i = threadIdx.x; i < total_dl_matrix_row_num[k]; i = i + blockDim.x)
		{
			for (int j = 0; j < vertex_num[k]; j++)
			{
				row_group[offset_row[k]+i]+= dl_matrix[offset_matrix[k] + i*total_dl_matrix_col_num[k]+j] * (j + 1);
			}
		}
	}

}

/* removed
__global__ void select_row(int* deleted_rows, int* row_group, int* search_depth, int *total_dl_matrix_row_num, int* selected_row_id)
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

__global__ void recover_deleted_rows(int* deleted_rows, int *search_depth, int *total_dl_matrix_row_num, int* offset_row, const int graph_count)
{
	for (int k = blockIdx.x; k<graph_count; k = k+ gridDim.x)
	{
		for (int i = threadIdx.x; i < total_dl_matrix_row_num[k]; i=i+blockDim.x)
		{
			if (abs(deleted_rows[offset_row[k]+i]) > search_depth[k] || deleted_rows[offset_row[k]+i] == search_depth[k])
			{
				deleted_rows[offset_row[k]+i] = 0;
			}
		}
	}

}

__global__ void recover_deleted_cols(int* deleted_cols, int *search_depth, int* total_dl_matrix_col_num, int *offset_col, const int graph_count)
{


	for (int k = blockIdx.x; k<graph_count; k = k+ gridDim.x)
	{
		for (int i = threadIdx.x; i < total_dl_matrix_col_num[k]; i=i+blockDim.x)
		{
			if (deleted_cols[offset_col[k]+i] >= search_depth[k])
			{
				deleted_cols[offset_col[k]+i] = 0;
			}
		}
	}

}

__global__ void recover_results(int* results, int *search_depth, int *total_dl_matrix_row_num, int *offset_row, const int graph_count)
{

	for (int k = blockIdx.x; k<graph_count; k = k+ gridDim.x)
	{
		for (int i = threadIdx.x; i < total_dl_matrix_row_num[k]; i = i + blockDim.x)
		{
			if (results[offset_row[k]+i] == search_depth[k])
			{
				results[offset_row[k]+i] = 0;
			}
		}
	}

}




__global__ void get_conflict_node_id(int* deleted_rows, int* row_group, int *search_depth, int *conflict_node_id, int *total_dl_matrix_row_num, int * offset_row, const int graph_count) 
{


	for (int k = blockIdx.x; k<graph_count; k = k+ gridDim.x)
	{
		for (int i = threadIdx.x; i < total_dl_matrix_row_num[k]; i = i+ blockDim.x) {
			if (row_group[offset_row[k]+i] == search_depth[k] + 1 && deleted_rows[offset_row[k]+i] > conflict_node_id[k]) {
				atomicExch(conflict_node_id+k, deleted_rows[offset_row[k]+i]);
			}
		}
	}

	//__syncthreads();
}


//problem
__global__ void get_conflict_col(int* dl_matrix, int* deleted_rows, int* deleted_cols, int* row_group, int *conflict_node_id, int *search_depth, int *conflict_col_id, int *vertex_num, int *total_dl_matrix_row_num, int *total_dl_matrix_col_num, int * offset_col, int * offset_row, int * offset_matrix, const int graph_count) 
{
	//*conflict_col_id = 0;
	for (int k = blockIdx.x; k<graph_count; k = k+ gridDim.x)
	{
		for (int i = threadIdx.x; i < total_dl_matrix_row_num[k]; i=i+blockDim.x) {  
			//find the conflict edge that connects current node and the most closest node.
			if (row_group[offset_row[k]+i] == search_depth[k] + 1 && deleted_rows[offset_row[k]+i] == conflict_node_id[k]) {
				for (int j = total_dl_matrix_col_num[k] - 1; j > vertex_num[k]; j--) {
					if (dl_matrix[offset_matrix[k]+i*total_dl_matrix_col_num[k]+j] * deleted_cols[offset_col[k]+j] == conflict_node_id[k]) {
						atomicExch(conflict_col_id+k, j);
					}
				}
			}
		}
	}

	__syncthreads();
}


__global__ void remove_cols(int* deleted_cols, int* col_group, int *conflict_col_id, int *total_dl_matrix_col_num, int *offset_col, const int graph_count)
{

	for (int k = blockIdx.x; k<graph_count; k = k+ gridDim.x)
	{
		for (int i = threadIdx.x; i < total_dl_matrix_col_num[k]; i = i + blockDim.x)
		{
			if (col_group[offset_col[k]+i] == col_group[conflict_col_id[k]])
			{
				deleted_cols[offset_col[k]+i] = -1;
			}
		}
	}

}

//+=============================================================================================================================================
__global__ void print_vec(int *vec, int start, int vec_length)
{
	for(int i=0; i<vec_length; i++)
	{
		printf("%d ", vec[start+i]);
	}
	printf("\n");
}

/*
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
*/
__global__ void add_gpu(int *vector, int idx, int val)
{
	vector[idx] = vector[idx] + val
}

__global__ void set_vector_value(int* vector, int idx, int val)
{
	vector[idx]=val;
}



void mc_solver(int* dl_matrix,	int* results, int* deleted_cols, int* deleted_rows,	int* col_group,	int* row_group,	int* conflict_count, int *vertex_num, int *total_dl_matrix_row_num, int *total_dl_matrix_col_num, int * offset_col, int * offset_row, int * offset_matrix, const int graph_count)
{
	//to be refreshed if one conflict reaches many counts
	int *search_depth = new int [graph_count];
	int *selected_row_id = new int [graph_count];
	int *vertex_num  = new int [graph_count];
	int *total_dl_matrix_col_num = new int [graph_count];
	int *total_dl_matrix_row_num= new int [graph_count];
	int *current_conflict_count= new int [graph_count];
	int *conflict_node_id= new int [graph_count];
	int *conflict_col_id= new int [graph_count];
	int *existance_of_candidate_rows=new int[graph_count];
	const int hard_conflict_threshold = 500;
	

	int *existance_of_candidate_rows_gpu;
	int *conflict_col_id_gpu;
	int *selected_row_id_gpu;
	int *conflict_node_id_gpu;
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
		cudaMemset(selected_row_id_gpu,-1,sizeof(int));
		//existance_of_candidate_rows=0;
		//selected_row_id=-1;
		check_existance_of_candidate_rows <<<block_count, thread_count >>> (deleted_rows, row_group, search_depth, existance_of_candidate_rows_gpu, total_dl_matrix_row_num);
		//__syncthreads();
		cudaMemcpy(existance_of_candidate_rows, existance_of_candidate_rows_gpu, sizeof(int), cudaMemcpyDeviceToHost);
		std::cout<<"check_existance_of_candidate_rows "<<std::endl;
		if (*existance_of_candidate_rows==1)
		{                                                                                                 //check if there are candidate rows existing, if no, do backtrace
			select_row <<<block_count, thread_count >>> (deleted_rows, row_group, search_depth, total_dl_matrix_row_num, selected_row_id_gpu); //select row and add to results
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
				add_gpu<<<1,1>>>(&deleted_rows[*selected_row_id],1);
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
				get_largest_value<<<block_count, thread_count >>>(conflict_count, conflict_col_id_gpu, total_dl_matrix_col_num);
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
