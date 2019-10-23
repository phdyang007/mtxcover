/**
 * @file   BitMatrix.h
 * @author Yibo Lin
 * @date   Oct 2019
 */

#pragma once

#include "cuda_runtime.h"

namespace gpu_mg {

/// bits from left to right as small to large 
template <typename T, unsigned int MaxNumBits>
struct BitArray
{
    static constexpr unsigned int unit_storage_size = sizeof(T)*8; ///< number of bits in unit storage T 
    static constexpr unsigned int num_units = (MaxNumBits + unit_storage_size - 1)/unit_storage_size; ///< number of unit Ts 

    T bits[num_units];

    __device__ bool at(unsigned int i) const 
    {
        unsigned int index = i/unit_storage_size; 
        unsigned int shift = i-index*unit_storage_size; 
        return (bits[index] >> (unit_storage_size-shift-1)) & T(1);
    }

    template <typename V>
    __host__ __device__ void init(const V* values, unsigned int n)
    {
        for (unsigned int i = 0; i < n; ++i)
        {
            set(i, values[i]); 
        }
    }

    __host__ __device__ void copy(const BitArray& rhs)
    {
        for (unsigned int i = 0; i < num_units; ++i)
        {
            bits[i] = rhs.bits[i]; 
        }
    }

    __host__ __device__ void set(unsigned int i, bool value) 
    {
        unsigned int index = i/unit_storage_size; 
        unsigned int shift = i-index*unit_storage_size; 
        // set to 0 first  
        T mask = (T(1) << (unit_storage_size-shift-1));
        bits[index] &= ~mask; 
        // set to value 
        mask = (T(value) << (unit_storage_size-shift-1));
        bits[index] |= mask; 
    }

    __host__ __device__ BitArray& operator&=(const BitArray& rhs)
    {
        for (unsigned int i = 0; i < num_units; ++i)
        {
            bits[i] &= rhs.bits[i]; 
        }
        return *this;
    }
    __host__ __device__ BitArray operator&(const BitArray& rhs) const
    {
        BitArray result; 
        result.copy(*this);
        result &= rhs; 
        return result;
    }
    __device__ BitArray& atomicAnd(const BitArray& rhs)
    {
        for (unsigned int i = 0; i < num_units; ++i)
        {
            atomicAnd(bits+i, rhs.bits[i]); 
        }
        return *this;
    }
    __host__ __device__ BitArray& operator|=(const BitArray& rhs)
    {
        for (unsigned int i = 0; i < num_units; ++i)
        {
            bits[i] |= rhs.bits[i]; 
        }
        return *this;
    }
    __host__ __device__ BitArray operator|(const BitArray& rhs) const
    {
        BitArray result; 
        result.copy(*this);
        result |= rhs; 
        return result;
    }
    __device__ BitArray& atomicOr(const BitArray& rhs)
    {
        for (unsigned int i = 0; i < num_units; ++i)
        {
            atomicOr(bits+i, rhs.bits[i]); 
        }
        return *this;
    }
    __host__ __device__ BitArray& operator^=(const BitArray& rhs)
    {
        for (unsigned int i = 0; i < num_units; ++i)
        {
            bits[i] ^= rhs.bits[i]; 
        }
        return *this;
    }
    __host__ __device__ BitArray operator^(const BitArray& rhs) const
    {
        BitArray result; 
        result.copy(*this);
        result ^= rhs; 
        return result;
    }
    __device__ BitArray& atomicXor(const BitArray& rhs)
    {
        for (unsigned int i = 0; i < num_units; ++i)
        {
            atomicXor(bits+i, rhs.bits[i]); 
        }
        return *this;
    }
    __host__ __device__ BitArray operator~() const
    {
        BitArray result; 
        for (unsigned int i = 0; i < num_units; ++i)
        {
            result.bits[i] = ~bits[i];
        }
        return result;
    }
};

template <typename T, unsigned int MaxColBits>
struct BitMatrices
{
    typedef BitArray<T, MaxColBits> bitarray_type; 

    bitarray_type* matrices; ///< rows of all matrices 
    unsigned int* matrix_row_offsets; ///< length of num_matrices+1, start indices in matrices 
    unsigned int* matrix_col_nums; ///< number of columns for each matrix 
    unsigned int num_matrices; ///< number of matrices 

    template <typename V>
    __host__ void init(const V* host_matrices, int* host_offset_matrix, int* host_total_dl_matrix_row_num, int* host_total_dl_matrix_row_num, int n)
    {
        num_matrices = n;

        int total_num_rows = std::accumulate(host_total_dl_matrix_row_num, host_total_dl_matrix_row_num+n, 0);

        std::vector<int> host_matrix_row_offsets (n+1);
        std::vector<bitarray_type> host_bit_matrices (total_num_rows);

        int count = 0; 
        for (int i = 0; i < n; ++i)
        {
            host_matrix_row_offsets[i] = count;

            int row_num = host_total_dl_matrix_row_num[i]; 
            int col_num = host_total_dl_matrix_col_num[i];
            const V* host_matrix = host_matrices+host_offset_matrix[i]; 
            for (int j = 0; j < row_num; ++j)
            {
                host_bit_matrices[count].init(host_matrix+j*col_num, col_num);
                ++count;
            }
        }
        host_matrix_row_offsets[n] = total_num_rows; 

        cudaMalloc((void*)&matrices, sizeof(bitarray_type)*total_num_rows);
        cudaMalloc((void*)&matrix_row_offsets, sizeof(unsigned int)*(num_matrices+1));
        cudaMalloc((void*)&matrix_col_nums, sizeof(unsigned int)*num_matrices);

        cudaMemcpy(matrices, host_bit_matrices, sizeof(bitarray_type)*total_num_rows, cudaMemcpyHostToDevice);
        cudaMemcpy(matrix_row_offsets, host_matrix_row_offsets, sizeof(unsigned int)*(num_matrices+1), cudaMemcpyHostToDevice);
        cudaMemcpy(matrix_col_nums, host_total_dl_matrix_col_num, sizeof(unsigned int)*n, cudaMemcpyHostToDevice);
    }

    __host__ void destroy()
    {
        cudaFree(matrices); 
        cudaFree(matrix_row_offsets); 
        cudaFree(matrix_col_nums);
    }

    /// @brief Get the start row for a matrix 
    __device__ const bitarray_type* matrix(unsigned int mat_id) const 
    {
#ifdef DEBUG
        assert(mat_id < num_matrices);
#endif
        return matrices[matrix_row_offsets[mat_id]];
    }
    /// @brief Get the start row for a matrix 
    __device__ bitarray_type* matrix(unsigned int mat_id) 
    {
#ifdef DEBUG
        assert(mat_id < num_matrices);
#endif
        return matrices[matrix_row_offsets[mat_id]];
    }

    /// @brief Get the row in a matrix 
    __device__ const bitarray_type& row(unsigned int mat_id, unsigned int row_id) const 
    {
#ifdef DEBUG
        assert(row_id < matrix_row_offsets[mat_id+1]-matrix_row_offsets[mat_id]);
#endif
        return matrix(mat_id)[row_id];
    }
    /// @brief Get the row in a matrix 
    __device__ bitarray_type& row(unsigned int mat_id, unsigned int row_id) 
    {
#ifdef DEBUG
        assert(row_id < matrix_row_offsets[mat_id+1]-matrix_row_offsets[mat_id]);
#endif
        return matrix(mat_id)[row_id];
    }
};

} // namespace gpu_mg
