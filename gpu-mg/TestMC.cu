#include "MatrixCoverGPU.cuh"
#include <iostream>
#include <fstream> 
#include <cstdio>







int main()
{
    std::ifstream file("matrix.txt");
    int graph_count=2;

    int total_dl_matrix_row_num[2]={276,276};

    int total_dl_matrix_col_num[2]={29,29};
    int offset_matrix[2]={0, 276*29};
    int offset_row[2]={0,276};
    int offset_col[2]={0,29};
    int *dl_matrix;
    std::cout<<"reading dl matrix from file"<<std::endl;
    int total_matrix = total_dl_matrix_row_num[0]*total_dl_matrix_col_num[0]+total_dl_matrix_row_num[1]*total_dl_matrix_col_num[1];
    int total_row = total_dl_matrix_row_num[0]+total_dl_matrix_row_num[1];
    int total_col = total_dl_matrix_col_num[0]+total_dl_matrix_col_num[1];
    dl_matrix = new int [total_matrix];

    std::cout<<"allocate dl matrix space from main memory"<<std::endl;
    for (int k = 0; k<graph_count; k++)
    {
        for (int i = 0; i < total_dl_matrix_row_num[k]; i++)
        {
            for (int j = 0; j < total_dl_matrix_col_num[k]; j++)
            {
    
                file>>dl_matrix[offset_matrix[k]+i*total_dl_matrix_col_num[k]+j];
            }
     
        }
    }

    std::cout<<"reading dl matrix from file  DONE"<<std::endl;


  

	std::ifstream col("col.txt");
    //int *deleted_cols = new int[total_dl_matrix_col_num];
    int *col_group = new int[total_col];
    for (int k =0; k<graph_count;k++){
        for(int i=0; i<total_dl_matrix_col_num[k]; i++){
            col>>col_group[offset_col[k]+i];
        }
    }

    int *deleted_cols = new int[total_col];
    int *deleted_rows = new int[total_row];

    int conflict_count[2] = {0,0};
    int vertex_num[2]= {5,5};


    //int *results = new int[total_dl_matrix_row_num];

    

    //allocate necessary vectors and matrix on GPU
    int *dl_matrix_gpu;
    int *results_gpu;
    cudaMalloc(&dl_matrix_gpu, sizeof(int*)*total_matrix);
    cudaMemcpy(dl_matrix_gpu, dl_matrix,  sizeof(int*)*total_matrix, cudaMemcpyHostToDevice);
    cudaMalloc(&results_gpu, sizeof(int*)*total_row);

    int *deleted_cols_gpu;
    int *deleted_rows_gpu;
    int *col_group_gpu;
    int *row_group_gpu;
    int *conflict_count_gpu;

    cudaMalloc(&deleted_cols_gpu, sizeof(int*)*total_col);
    cudaMalloc(&deleted_rows_gpu, sizeof(int*)*total_row);
    cudaMalloc(&col_group_gpu, sizeof(int*)*total_col);
    cudaMalloc(&row_group_gpu, sizeof(int*)*total_row);
    cudaMalloc(&conflict_count_gpu, sizeof(int*)*total_col);
    cudaMemcpy(col_group_gpu, col_group,  sizeof(int*)*total_col, cudaMemcpyHostToDevice);


    int *vertex_num_gpu;
    int *total_dl_matrix_col_num_gpu;
    int *total_dl_matrix_row_num_gpu;


    cudaMalloc(&vertex_num_gpu, sizeof(int*)*graph_count);
    cudaMalloc(&total_dl_matrix_col_num_gpu, sizeof(int*)*graph_count);
    cudaMalloc(&total_dl_matrix_row_num_gpu, sizeof(int*)*graph_count);

    cudaMemcpy(vertex_num_gpu, vertex_num,  sizeof(int*)*graph_count, cudaMemcpyHostToDevice);
    cudaMemcpy(total_dl_matrix_col_num_gpu, total_dl_matrix_col_num,  sizeof(int*)*graph_count, cudaMemcpyHostToDevice);
    cudaMemcpy(total_dl_matrix_row_num_gpu, total_dl_matrix_row_num,  sizeof(int*)*graph_count, cudaMemcpyHostToDevice);


    int *offset_col_gpu;
    int *offset_row_gpu;
    int *offset_matrix_gpu;


    cudaMalloc(&offset_col_gpu, sizeof(int*)*graph_count);
    cudaMalloc(&offset_row_gpu, sizeof(int*)*graph_count);
    cudaMalloc(&offset_matrix_gpu, sizeof(int*)*graph_count);

    cudaMemcpy(offset_col_gpu, offset_col,  sizeof(int*)*graph_count, cudaMemcpyHostToDevice);
    cudaMemcpy(offset_row_gpu, offset_row,  sizeof(int*)*graph_count, cudaMemcpyHostToDevice);
    cudaMemcpy(offset_matrix_gpu, offset_matrix,  sizeof(int*)*graph_count, cudaMemcpyHostToDevice);

    int *search_depth_gpu;
    int *selected_row_id_gpu;
    int *current_conflict_count_gpu;
    int *conflict_node_id_gpu;
    int *conflict_col_id_gpu;
    int *existance_of_candidate_rows_gpu;


    cudaMalloc(&search_depth_gpu, sizeof(int*)*graph_count);
    cudaMalloc(&selected_row_id_gpu, sizeof(int*)*graph_count);
    cudaMalloc(&current_conflict_count_gpu, sizeof(int*)*graph_count);
    cudaMalloc(&conflict_node_id_gpu, sizeof(int*)*graph_count);
    cudaMalloc(&conflict_col_id_gpu, sizeof(int*)*graph_count);
    cudaMalloc(&existance_of_candidate_rows_gpu, sizeof(int*)*graph_count);
    

    int hard_conflict_threshold=500;

    //int * row_group=new int[total_dl_matrix_row_num];
    //get col and row group
    //init_vectors<<<1, 32>>>(row_group_gpu, total_dl_matrix_row_num_gpu);
    //init_vectors<<<1, 32>>>(deleted_cols_gpu, total_dl_matrix_col_num_gpu);
    //init_vectors<<<1, 32>>>(deleted_rows_gpu, total_dl_matrix_row_num_gpu);
    //init_vectors<<<1, 32>>>(results_gpu, total_dl_matrix_row_num_gpu);
    //init_vectors<<<1, 32>>>(conflict_count_gpu, total_dl_matrix_col_num_gpu);
    //init_vectors<<<1, 32>>>(deleted_rows_gpu, total_dl_matrix_row_num_gpu);
    //cudaDeviceSynchronize();
    //cudaMemcpy(row_group, row_group_gpu, sizeof(int)*total_dl_matrix_row_num_gpu, cudaMemcpyDeviceToHost);
    //std::cout<<"print row group"<<std::endl;
    //for(int i=0; i<total_dl_matrix_row_num; i++)
    //{
    //    std::cout<<row_group[i]<<' ';
    //}
    //std::cout<<std::endl;

    //get_vertex_row_group<<<1, 32>>>(row_group_gpu, dl_matrix_gpu, vertex_num_gpu, total_dl_matrix_row_num_gpu, total_dl_matrix_col_num_gpu);
    //cudaMemcpy(row_group, row_group_gpu, sizeof(int)*total_dl_matrix_row_num_gpu, cudaMemcpyDeviceToHost);
    //std::cout<<"print row group"<<std::endl;
    //for(int i=0; i<total_dl_matrix_row_num; i++)
   // {
     //   std::cout<<row_group[i]<<' ';
    //}
    //std::cout<<std::endl;
    //cudaMemcpy(col_group_gpu, col_group, sizeof(int)*total_dl_matrix_col_num, cudaMemcpyHostToDevice);


    //delete_rows_and_columns<<<1, 32>>>(dl_matrix_gpu, deleted_rows_gpu, deleted_cols_gpu, 1, 1, total_dl_matrix_row_num, total_dl_matrix_col_num);


    //cudaMemcpy(deleted_cols, deleted_cols_gpu, sizeof(int)*total_dl_matrix_col_num, cudaMemcpyDeviceToHost);
    //cudaMemcpy(deleted_rows, deleted_rows_gpu, sizeof(int)*total_dl_matrix_row_num, cudaMemcpyDeviceToHost);
    //print_vec(deleted_cols,total_dl_matrix_col_num);
    //print_vec(deleted_rows,total_dl_matrix_row_num);

    //print_vec_g<<<1,1>>>(col_group_gpu, total_col);



    cudaDeviceSynchronize();
    gpu_mg::mc_solver<<<2,32>>>(dl_matrix_gpu, results_gpu, 
        deleted_cols_gpu, deleted_rows_gpu, col_group_gpu, row_group_gpu, conflict_count_gpu,
        vertex_num_gpu, total_dl_matrix_row_num_gpu, total_dl_matrix_col_num_gpu,
        offset_col_gpu, offset_row_gpu, offset_matrix_gpu,
        search_depth_gpu, selected_row_id_gpu, current_conflict_count_gpu, conflict_node_id_gpu, conflict_col_id_gpu, existance_of_candidate_rows_gpu,
        graph_count, hard_conflict_threshold);
    cudaDeviceSynchronize();
    //mc_solver(dl_matrix_gpu, results_gpu, deleted_cols_gpu, deleted_rows_gpu, col_group_gpu, row_group_gpu, conflict_count_gpu, vertex_num_gpu, total_dl_matrix_row_num_gpu, total_dl_matrix_col_num_gpu);
    std::cout<<"================================================================================================================================="<<std::endl;

    int *results = new int [total_row];
    cudaMemcpy(results, results_gpu, sizeof(int)*total_row, cudaMemcpyDeviceToHost);
    cudaMemcpy(deleted_cols, deleted_cols_gpu, sizeof(int)*total_col, cudaMemcpyDeviceToHost);
    for (int k=0; k < graph_count; k++)
    {    
        for (int i = 0; i < total_dl_matrix_row_num[k]; i++)
        {
            std::cout << results[offset_row[k]+i] << ' ';
        }
        std::cout << std::endl;
        for (int i = 0; i < total_dl_matrix_row_num[k]; i++){
            if(results[offset_row[k]+i]>0){
                std::cout << i << ' '; 
            }
        }
        std::cout<<std::endl;
        for (int i = 0; i < total_dl_matrix_col_num[k]; i++)
        {
            if (deleted_cols[offset_col[k]+i] == -1)
            {
                conflict_count[k]++;
            }
        }
    
        std::cout << "Conflict Num is " << conflict_count[k] / 3 << std::endl;

    }

       
    

	

    cudaFree(dl_matrix_gpu);
    cudaFree(results_gpu);
    cudaFree(deleted_cols_gpu);
    cudaFree(deleted_rows_gpu);
    cudaFree(col_group_gpu);
    cudaFree(row_group_gpu);
    cudaFree(conflict_count_gpu);
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


    delete[] results;

    delete[] dl_matrix;


    //delete[] test_matrix;
    delete[] deleted_cols;
    delete[] deleted_rows;
    //delete[] gtcol;
	delete[] col_group;
	//delete[] gtr;
    return 0;
}
