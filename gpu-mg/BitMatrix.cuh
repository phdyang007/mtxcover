/**
 * @file   BitMatrix.h
 * @author Yibo Lin
 * @date   Oct 2019
 */

#pragma once

#include <numeric>
#include "cuda_runtime.h"

namespace gpu_mg {

/// bits from left to right as small to large 
template <typename T, unsigned int MaxNumBits>
struct BitArray
{
    typedef T type; 

    static_assert(std::is_unsigned<T>::value);

    static constexpr unsigned int unit_storage_size = sizeof(T)*8; ///< number of bits in unit storage T 
    static constexpr unsigned int num_units = (MaxNumBits + unit_storage_size - 1)/unit_storage_size; ///< number of unit Ts 
    static constexpr T all_ones = ~(T)0; ///< an integer sets all bits to 1
    
    T bits[num_units];

    __device__ bool at(unsigned int i) const 
    {
        unsigned int index = i/unit_storage_size; 
        unsigned int shift = i-index*unit_storage_size; 
        return (bits[index] >> (unit_storage_size-shift-1)) & T(1);
    }

    __host__ __device__ void reset()
    {
        for (unsigned int i = 0; i < num_units; ++i)
        {
            bits[i] = 0; 
        }
    }
    /// Users need to know the implementation and make sure there is no data race 
    __device__ void block_reset()
    {
        for (unsigned int i = threadIdx.x; i < num_units; i += blockDim.x)
        {
            bits[i] = 0; 
        }
    }
    __host__ __device__ void set()
    {
        for (unsigned int i = 0; i < num_units; ++i)
        {
            bits[i] = all_ones; 
        }
    }
    /// Users need to know the implementation and make sure there is no data race 
    __host__ __device__ void block_set()
    {
        for (unsigned int i = threadIdx.x; i < num_units; i += blockDim.x)
        {
            bits[i] = all_ones; 
        }
    }

    template <typename V>
    __host__ __device__ void init(const V* values, unsigned int n)
    {
        for (unsigned int i = 0; i < n; ++i)
        {
            if (values[i])
            {
                set(i); 
            }
            else 
            {
                reset(i);
            }
        }
    }

    __host__ __device__ void copy(const BitArray& rhs)
    {
        for (unsigned int i = 0; i < num_units; ++i)
        {
            bits[i] = rhs.bits[i]; 
        }
    }
    __device__ void block_copy(const BitArray& rhs)
    {
        for (unsigned int i = threadIdx.x; i < num_units; i += blockDim.x)
        {
            bits[i] = rhs.bits[i]; 
        }
    }

    __host__ __device__ void reset(unsigned int i) 
    {
        unsigned int index = i/unit_storage_size; 
        unsigned int shift = i-index*unit_storage_size; 
        // set to 0 
        T mask = ~(T(1) << (unit_storage_size-shift-1));
        bits[index] &= mask;
    }
    __host__ __device__ void set(unsigned int i) 
    {
        unsigned int index = i/unit_storage_size; 
        unsigned int shift = i-index*unit_storage_size; 
        // set to 1 
        T mask = (T(1) << (unit_storage_size-shift-1));
        bits[index] |= mask; 
    }
    /// all the atomic operators here cannot be mixed in usage. 
    /// For example, one thread is using atomicSet to location i, 
    /// and another thread is using atomicReset to location i. 
    /// Then the results are undefined due to indeterministic order. 
    /// But I do guarantee atomicSet itself is atomic. 
    __host__ __device__ void atomic_reset(unsigned int i) 
    {
        unsigned int index = i/unit_storage_size; 
        unsigned int shift = i-index*unit_storage_size; 

        // set to 0
        T mask = ~(T(1) << (unit_storage_size-shift-1));
        atomicAnd(bits+index, mask);
    }
    __host__ __device__ void atomic_set(unsigned int i) 
    {
        unsigned int index = i/unit_storage_size; 
        unsigned int shift = i-index*unit_storage_size; 

        // set to 1
        T mask = (T(1) << (unit_storage_size-shift-1));
        atomicOr(bits+index, mask);
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
    /// Users need to know the implementation and make sure there is no data race 
    __device__ BitArray& block_and(const BitArray& rhs)
    {
        for (unsigned int i = threadIdx.x; i < num_units; i += blockDim.x)
        {
            bits[i] &= rhs.bits[i]; 
        }
        return *this;
    }
    __device__ BitArray& atomic_and(const BitArray& rhs)
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
    /// Users need to know the implementation and make sure there is no data race 
    __device__ BitArray& block_or(const BitArray& rhs)
    {
        for (unsigned int i = threadIdx.x; i < num_units; i += blockDim.x)
        {
            bits[i] |= rhs.bits[i];
        }
        return *this;
    }
    __device__ BitArray& atomic_or(const BitArray& rhs)
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
    __device__ BitArray& atomic_xor(const BitArray& rhs)
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

template <typename T, unsigned int MaxRowBits, unsigned int MaxColBits>
class BitMatrices
{
    public:
        typedef BitArray<T, MaxColBits> row_type; 
        typedef BitArray<T, MaxRowBits> col_type; 

        template <typename V>
        __host__ void init(const V* host_matrices, int* host_offset_matrix, int* host_total_dl_matrix_row_num, int* host_total_dl_matrix_col_num, int n)
        {
            m_num_matrices = n;

            int total_num_rows = std::accumulate(host_total_dl_matrix_row_num, host_total_dl_matrix_row_num+n, 0);

            std::vector<unsigned int> host_matrix_row_offsets (n+1);
            std::vector<row_type> host_rows (total_num_rows);

            int count = 0; 
            for (int i = 0; i < n; ++i)
            {
                host_matrix_row_offsets[i] = count;

                int row_num = host_total_dl_matrix_row_num[i]; 
                int col_num = host_total_dl_matrix_col_num[i];
                const V* host_matrix = host_matrices+host_offset_matrix[i]; 
                for (int j = 0; j < row_num; ++j)
                {
                    host_rows[count].init(host_matrix+j*col_num, col_num);
                    ++count;
                }
            }
            host_matrix_row_offsets[n] = total_num_rows; 

            int total_num_cols = std::accumulate(host_total_dl_matrix_col_num, host_total_dl_matrix_col_num+n, 0);
            std::vector<unsigned int> host_matrix_col_offsets (n+1);
            std::vector<col_type> host_cols (total_num_cols);
            std::vector<int> host_col; 

            count = 0; 
            for (int i = 0; i < n; ++i)
            {
                host_matrix_col_offsets[i] = count;

                int row_num = host_total_dl_matrix_row_num[i]; 
                int col_num = host_total_dl_matrix_col_num[i];
                const V* host_matrix = host_matrices+host_offset_matrix[i]; 
                host_col.resize(row_num); 
                for (int j = 0; j < col_num; ++j)
                {
                    for (int k = 0; k < row_num; ++k)
                    {
                        host_col[k] = host_matrix[k*col_num + j];
                    }
                    host_cols[count].init(host_col.data(), row_num);
                    ++count;
                }
            }
            host_matrix_col_offsets[n] = total_num_cols;

            cudaMalloc((void**)&m_rows, sizeof(row_type)*total_num_rows);
            cudaMalloc((void**)&m_matrix_row_offsets, sizeof(unsigned int)*(n+1));
            cudaMalloc((void**)&m_cols, sizeof(col_type)*total_num_cols);
            cudaMalloc((void**)&m_matrix_col_offsets, sizeof(unsigned int)*(n+1));

            cudaMemcpy(m_rows, host_rows.data(), sizeof(row_type)*total_num_rows, cudaMemcpyHostToDevice);
            cudaMemcpy(m_matrix_row_offsets, host_matrix_row_offsets.data(), sizeof(unsigned int)*(n+1), cudaMemcpyHostToDevice);
            cudaMemcpy(m_cols, host_cols.data(), sizeof(col_type)*total_num_cols, cudaMemcpyHostToDevice);
            cudaMemcpy(m_matrix_col_offsets, host_matrix_col_offsets.data(), sizeof(unsigned int)*(n+1), cudaMemcpyHostToDevice);
        }

        __host__ void destroy()
        {
            cudaFree(m_rows); 
            cudaFree(m_matrix_row_offsets); 
            cudaFree(m_cols); 
            cudaFree(m_matrix_col_offsets); 
        }

        /// @brief Get the start row for a matrix 
        __device__ const row_type* rows(unsigned int mat_id) const 
        {
#ifdef DEBUG
            assert(mat_id < m_num_matrices);
#endif
            return m_rows + m_matrix_row_offsets[mat_id];
        }
        /// @brief Get the start row for a matrix 
        __device__ row_type* rows(unsigned int mat_id) 
        {
#ifdef DEBUG
            assert(mat_id < m_num_matrices);
#endif
            return m_rows + m_matrix_row_offsets[mat_id];
        }

        /// @brief Get the row in a matrix 
        __device__ const row_type& row(unsigned int mat_id, unsigned int row_id) const 
        {
#ifdef DEBUG
            assert(row_id < m_matrix_row_offsets[mat_id+1]-m_matrix_row_offsets[mat_id]);
#endif
            return rows(mat_id)[row_id];
        }
        /// @brief Get the row in a matrix 
        __device__ row_type& row(unsigned int mat_id, unsigned int row_id) 
        {
#ifdef DEBUG
            assert(row_id < m_matrix_row_offsets[mat_id+1]-m_matrix_row_offsets[mat_id]);
#endif
            return rows(mat_id)[row_id];
        }

        /// @brief Get the start col for a matrix 
        __device__ const col_type* cols(unsigned int mat_id) const 
        {
#ifdef DEBUG
            assert(mat_id < m_num_matrices);
#endif
            return m_cols + m_matrix_col_offsets[mat_id];
        }
        /// @brief Get the start col for a matrix 
        __device__ col_type* cols(unsigned int mat_id) 
        {
#ifdef DEBUG
            assert(mat_id < m_num_matrices);
#endif
            return m_cols + m_matrix_col_offsets[mat_id];
        }

        /// @brief Get the col in a matrix 
        __device__ const col_type& col(unsigned int mat_id, unsigned int col_id) const 
        {
#ifdef DEBUG
            assert(col_id < m_matrix_col_offsets[mat_id+1]-m_matrix_col_offsets[mat_id]);
#endif
            return cols(mat_id)[col_id];
        }
        /// @brief Get the col in a matrix 
        __device__ col_type& col(unsigned int mat_id, unsigned int col_id) 
        {
#ifdef DEBUG
            assert(col_id < m_matrix_col_offsets[mat_id+1]-m_matrix_col_offsets[mat_id]);
#endif
            return cols(mat_id)[col_id];
        }

    protected:
        row_type* m_rows; ///< rows of all matrices 
        col_type* m_cols; ///< columns of all transposed matrices 
        unsigned int* m_matrix_row_offsets; ///< length of num_matrices+1, start indices in matrices 
        unsigned int* m_matrix_col_offsets; ///< length of num_matrices+1, start indices in transposed matrices 
        unsigned int m_num_matrices; ///< number of matrices 
};

} // namespace gpu_mg
