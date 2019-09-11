#include "MatrixCoverGPU.cuh"
#include <iostream>
#include <fstream> 
#include <cstdio>







int main()
{
	std::ifstream file("matrix.txt");
    int total_dl_matrix_row_num=276;

    int total_dl_matrix_col_num=29;

    int *dl_matrix;
    std::cout<<"reading dl matrix from file"<<std::endl;

    dl_matrix = new int [total_dl_matrix_row_num*total_dl_matrix_col_num];

    std::cout<<"allocate dl matrix space from main memory"<<std::endl;
    for (int i = 0; i < total_dl_matrix_row_num; i++)
    {
        for (int j = 0; j < total_dl_matrix_col_num; j++)
        {

            file>>dl_matrix[i*total_dl_matrix_col_num+j];
        }
 
    }
    std::cout<<"reading dl matrix from file  DONE"<<std::endl;


  

	std::ifstream col("col.txt");
    //int *deleted_cols = new int[total_dl_matrix_col_num];
    int *col_group = new int[total_dl_matrix_col_num];
    for(int i=0; i<total_dl_matrix_col_num; i++){
    	col>>col_group[i];
	}
    int *deleted_cols = new int[total_dl_matrix_col_num];
    int *deleted_rows = new int[total_dl_matrix_row_num];

    int conflict_count = 0;
    int vertex_num = 5;
    int vertex_num_gpu =vertex_num;
    int total_dl_matrix_row_num_gpu =total_dl_matrix_row_num;
    int total_dl_matrix_col_num_gpu =total_dl_matrix_col_num;
    //int *results = new int[total_dl_matrix_row_num];

    

    //allocate necessary vectors and matrix on GPU
    int *dl_matrix_gpu;
    int *deleted_cols_gpu;
    int *col_group_gpu;
    int *results_gpu;
    int* conflict_count_gpu;
	int* deleted_rows_gpu;
	int* row_group_gpu;
    //dl_matrix_gpu = new int *[total_dl_matrix_row_num];
    cudaMalloc(&dl_matrix_gpu, sizeof(int*)*total_dl_matrix_row_num*total_dl_matrix_col_num);
    std::cout<<"allocate space on GPU  DONE"<<std::endl;

    cudaMemcpy(dl_matrix_gpu, dl_matrix, sizeof(int)*total_dl_matrix_row_num*total_dl_matrix_col_num, cudaMemcpyHostToDevice);


    std::cout<<"copy dl matrix to device  DONE"<<std::endl;

    cudaMalloc(&deleted_cols_gpu, sizeof(int) * total_dl_matrix_col_num);
    cudaMalloc(&col_group_gpu, sizeof(int) * total_dl_matrix_col_num);
    cudaMalloc(&results_gpu, sizeof(int) * total_dl_matrix_row_num);
    cudaMalloc(&conflict_count_gpu, sizeof(int) * total_dl_matrix_col_num);
    cudaMalloc(&deleted_rows_gpu, sizeof(int) * total_dl_matrix_row_num);
    cudaMalloc(&row_group_gpu, sizeof(int) * total_dl_matrix_row_num);


    int * row_group=new int[total_dl_matrix_row_num];
    //get col and row group
    init_vectors<<<1, 32>>>(row_group_gpu, total_dl_matrix_row_num_gpu);
    //init_vectors<<<1, 32>>>(deleted_cols_gpu, total_dl_matrix_col_num_gpu);
    //init_vectors<<<1, 32>>>(deleted_rows_gpu, total_dl_matrix_row_num_gpu);
    //init_vectors<<<1, 32>>>(results_gpu, total_dl_matrix_row_num_gpu);
    //init_vectors<<<1, 32>>>(conflict_count_gpu, total_dl_matrix_col_num_gpu);
    //init_vectors<<<1, 32>>>(deleted_rows_gpu, total_dl_matrix_row_num_gpu);
    //cudaDeviceSynchronize();
    cudaMemcpy(row_group, row_group_gpu, sizeof(int)*total_dl_matrix_row_num_gpu, cudaMemcpyDeviceToHost);
    std::cout<<"print row group"<<std::endl;
    for(int i=0; i<total_dl_matrix_row_num; i++)
    {
        std::cout<<row_group[i]<<' ';
    }
    std::cout<<std::endl;

    get_vertex_row_group<<<1, 32>>>(row_group_gpu, dl_matrix_gpu, vertex_num_gpu, total_dl_matrix_row_num_gpu, total_dl_matrix_col_num_gpu);
    cudaMemcpy(row_group, row_group_gpu, sizeof(int)*total_dl_matrix_row_num_gpu, cudaMemcpyDeviceToHost);
    std::cout<<"print row group"<<std::endl;
    for(int i=0; i<total_dl_matrix_row_num; i++)
    {
        std::cout<<row_group[i]<<' ';
    }
    std::cout<<std::endl;
    cudaMemcpy(col_group_gpu, col_group, sizeof(int)*total_dl_matrix_col_num, cudaMemcpyHostToDevice);


    //delete_rows_and_columns<<<1, 32>>>(dl_matrix_gpu, deleted_rows_gpu, deleted_cols_gpu, 1, 1, total_dl_matrix_row_num, total_dl_matrix_col_num);


    //cudaMemcpy(deleted_cols, deleted_cols_gpu, sizeof(int)*total_dl_matrix_col_num, cudaMemcpyDeviceToHost);
    //cudaMemcpy(deleted_rows, deleted_rows_gpu, sizeof(int)*total_dl_matrix_row_num, cudaMemcpyDeviceToHost);
    //print_vec(deleted_cols,total_dl_matrix_col_num);
    //print_vec(deleted_rows,total_dl_matrix_row_num);
    mc_solver(dl_matrix_gpu, results_gpu, deleted_cols_gpu, deleted_rows_gpu, col_group_gpu, row_group_gpu, conflict_count_gpu, vertex_num_gpu, total_dl_matrix_row_num_gpu, total_dl_matrix_col_num_gpu);


    int *results = new int [total_dl_matrix_row_num];
    cudaMemcpy(results, results_gpu, sizeof(int)*total_dl_matrix_row_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(deleted_cols, deleted_cols_gpu, sizeof(int)*total_dl_matrix_col_num, cudaMemcpyDeviceToHost);

    for (int i = 0; i < total_dl_matrix_row_num; i++)
    {
        std::cout << results[i] << ' ';
    }
    std::cout << std::endl;
    for (int i = 0; i < total_dl_matrix_row_num; i++){
    	if(results[i]>0){
    		std::cout << i << ' '; 
		}
	}
	std::cout<<std::endl;
    for (int i = 0; i < total_dl_matrix_col_num; i++)
    {
        if (deleted_cols[i] == -1)
        {
            conflict_count++;
        }
    }

    std::cout << "Conflict Num is " << conflict_count / 3 << std::endl;
       
    
    std::cout<<"ground truth"<<std::endl;
    std::ifstream gt("gt.txt");
    int *gtr=new int[vertex_num];
    for(int i=0; i<vertex_num; i++){
    	gt>>gtr[i];
	}
	int *gtcol= new int[total_dl_matrix_col_num];
	init_vectors<<<1,32>>>(gtcol, total_dl_matrix_col_num);

	for(int i=0; i<vertex_num; i++){
		for(int j=0; j<total_dl_matrix_col_num; j++){
			if(gtr[i]<vertex_num*3){
				gtcol[j]=gtcol[j]+dl_matrix[i*total_dl_matrix_col_num+j];
			}else{
				gtcol[j]=gtcol[j]+dl_matrix[(i-1)*total_dl_matrix_col_num+j];
			}
		}
	}
	
	for(int i=0; i<total_dl_matrix_col_num; i++){
		std::cout << gtcol[i]<< ' ';
	}
	std::cout<< std::endl;
	

    cudaFree(dl_matrix_gpu);
    //delete [] dl_matrix_gpu;
    cudaFree(deleted_cols_gpu);
    cudaFree(col_group_gpu);
    cudaFree(results_gpu);
    cudaFree(conflict_count_gpu);
    cudaFree(deleted_rows_gpu);
    cudaFree(row_group_gpu);
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
