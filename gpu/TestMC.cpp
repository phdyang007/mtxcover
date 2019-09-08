#include "MatrixCoverGPU.cuh"
#include <iostream>
#include <fstream> 


int main()
{
	std::ifstream file("matrix.txt");
    int total_dl_matrix_row_num=2244;
    //int total_dl_matrix_row_num=235; /
    int total_dl_matrix_col_num=55;
	/*
    int tmp_matrix[total_dl_matrix_row_num][total_dl_matrix_col_num] = {
        {1, 0, 0, 1, 0, 1, 0, 0, 0},
        {1, 0, 0, 0, 1, 0, 1, 0, 1},
        {0, 1, 0, 1, 0, 0, 0, 1, 1},
        {0, 1, 0, 0, 1, 0, 0, 0, 1},
        {0, 0, 1, 0, 0, 1, 0, 1, 1},
        {0, 0, 1, 0, 0, 0, 1, 0, 1},
        {0, 0, 1, 0, 0, 1, 0, 0, 1},
        {0, 0, 1, 0, 0, 0, 1, 1, 1}};
	*/
    int **dl_matrix;

    dl_matrix = new int *[total_dl_matrix_row_num];
    for (int i = 0; i < total_dl_matrix_row_num; i++)
    {
        dl_matrix[i] = new int[total_dl_matrix_col_num];
        for (int j = 0; j < total_dl_matrix_col_num; j++)
        {
            file>>dl_matrix[i][j];
        }
    }




	std::ifstream col("col.txt");
    int *deleted_cols = new int[total_dl_matrix_col_num];
    int *col_group = new int[total_dl_matrix_col_num];
    for(int i=0; i<total_dl_matrix_col_num; i++){
    	col>>col_group[i];
	}



    //int col_group[total_dl_matrix_col_num]={0};// = {-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9 10 10 10 11 11 11 12 12 12 13 13 13 14 14 14 15 15 15 16 16 16 17 17 17 18 18 18 19 19 19 20 20 20 21 21 21};
	//memset(col_group, 0, sizeof(col_group));
    int conflict_count = 0;
    int vertex_num = 10;
    __shared__ int vertex_num_gpu =vertex_num;
    __shared__ int total_dl_matrix_row_num_gpu =total_dl_matrix_row_num;
    __shared__ int total_dl_matrix_col_num_gpu =total_dl_matrix_col_num;
    //int *results = new int[total_dl_matrix_row_num];

    int **dl_matrix_gpu;
    //int *dl_matrix_row_gpu;
    __shared__ int deleted_cols_gpu[total_dl_matrix_col_num];
    __shared__ int col_group_gpu[total_dl_matrix_col_num;
    __shared__ int results_gpu[total_dl_matrix_row_num];
    
    dl_matrix_gpu = new int *[total_dl_matrix_row_num];

    for(int i=0; i<total_dl_matrix_row_num; i++)
    {
        cudaMalloc(&dl_matrix_gpu[i], sizeof(int) * total_dl_matrix_col_num);
        cudaMemcpy(dl_matrix_gpu[i], dl_matrix[i], sizeof(int)*total_dl_matrix_col_num, cudaMemcpyHostToDevice);
    }


    cudaMemcpy(col_group_gpu, col_group, sizeof(int)*total_dl_matrix_col_num, cudaMemcpyHostToDevice);

    mc_solver<<<1,1024>>>(dl_matrix_gpu, results_gpu, deleted_cols_gpu, col_group_gpu, vertex_num_gpu, total_dl_matrix_row_num_gpu, total_dl_matrix_col_num_gpu);

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
   	/*
    std::ifstream gt("gt.txt");
    int *gtr=new int[vertex_num];
    for(int i=0; i<vertex_num; i++){
    	gt>>gtr[i];
	}
	int *gtcol= new int[total_dl_matrix_col_num];
	init_vectors(gtcol, total_dl_matrix_col_num, 0);

	for(int i=0; i<vertex_num; i++){
		for(int j=0; j<total_dl_matrix_col_num; j++){
			if(gtr[i]<vertex_num*3){
				gtcol[j]=gtcol[j]+dl_matrix[i][j];
			}else{
				gtcol[j]=gtcol[j]+dl_matrix[i-1][j];
			}
		}
	}
	
	for(int i=0; i<total_dl_matrix_col_num; i++){
		std::cout << gtcol[i]<< ' ';
	}
	std::cout<< std::endl;
	*/
    for(int i=0; i<total_dl_matrix_row_num; i++)
    {
        cudaFree(dl_matrix_gpu[i]);
    }
    delete [] dl_matrix_gpu;
    delete[] results;
    for (int i = 0; i < total_dl_matrix_row_num; i++)
    {
        delete[] dl_matrix[i];
    }
    delete[] dl_matrix;
    delete[] deleted_cols;
    //delete[] gtcol;
	delete[] col_group;
	//delete[] gtr;
    return 0;
}
