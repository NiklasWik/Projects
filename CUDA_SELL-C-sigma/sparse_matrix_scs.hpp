// 
// Author: Niklas Wik 
//

#ifndef sparse_matrix_scs_hpp
#define sparse_matrix_scs_hpp

#include <utility>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <omp.h>

#include <vector>

#include "vector.hpp"

#ifndef DISABLE_CUDA
template <typename Number>
__global__ void compute_spmv(const std::size_t n_rows,
                             const std::size_t *chunk_starts,
                             const std::size_t *chunk_widths,
                             const unsigned int *column_indices,
                             const Number *values, 
                             const Number *src,
                             Number *dst) {
    const unsigned int chunk = blockIdx.x;
    const unsigned int row = threadIdx.x;
    const unsigned int real_row = row + chunk * blockDim.x;
    const unsigned int cs = chunk_starts[chunk];

    Number sum = 0;
    for (unsigned int i = 0; i < chunk_widths[chunk]; i++) {
        sum += values[cs + row + i * blockDim.x] *
               src[column_indices[cs + row + i * blockDim.x]];
    }
    if (real_row < n_rows) dst[real_row] = sum;
}
#endif

// Sparse matrix in CELL-C-sigma format. (sigma = 1)

template <typename Number>
class SparseMatrix_SCS {
   public:
    static const int C = 32; // OBS, in order to run on CPU, 
                             // chunk size has to be C = 4!!!
    static const int block_size = Vector<Number>::block_size;

    SparseMatrix_SCS(const std::vector<unsigned int> &row_lengths,
                     const MemorySpace memory_space,
                     const MPI_Comm communicator)
        : communicator(communicator), memory_space(memory_space) {
        n_rows = row_lengths.size();
        n_chunks = (n_rows + C - 1) / C;  // All chunks have C rows.
        chunk_starts = new std::size_t[n_chunks + 1];
        chunk_widths = new std::size_t[n_chunks];

#pragma omp parallel for
        for (unsigned int chunk = 0; chunk < n_chunks; ++chunk) {
            chunk_starts[chunk] = 0;
            chunk_widths[chunk] = 0;
        }
        chunk_starts[n_chunks] = 0;

        for (unsigned int chunk = 0; chunk < n_chunks; chunk++) {
            for (unsigned int row = 0; row < C; row++) {
                if (chunk * C + row < n_rows)
                    if (row_lengths[chunk * C + row] > chunk_widths[chunk])
                        chunk_widths[chunk] = row_lengths[chunk * C + row];
            }
            chunk_starts[chunk + 1] =
                chunk_starts[chunk] + chunk_widths[chunk] * C;
        }

        n_entries = chunk_starts[n_chunks];

        if (memory_space == MemorySpace::CUDA) {
            std::size_t *host_chunk_starts = chunk_starts;
            chunk_starts = 0;
            AssertCuda(cudaMalloc(&chunk_starts,
                                  (n_chunks + 1) * sizeof(std::size_t)));
            AssertCuda(cudaMemcpy(chunk_starts, host_chunk_starts,
                                  (n_chunks + 1) * sizeof(std::size_t),
                                  cudaMemcpyHostToDevice));
            delete[] host_chunk_starts;

            std::size_t *host_chunk_widths = chunk_widths;
            chunk_widths = 0;
            AssertCuda(cudaMalloc(&chunk_widths, (n_chunks) * sizeof(std::size_t)));
            AssertCuda(cudaMemcpy(chunk_widths, host_chunk_widths,
                                  (n_chunks) * sizeof(std::size_t),
                                  cudaMemcpyHostToDevice));
            delete[] host_chunk_widths;

            AssertCuda(cudaMalloc(&column_indices, n_entries * sizeof(unsigned int)));
            AssertCuda(cudaMalloc(&values, n_entries * sizeof(Number)));

#ifndef DISABLE_CUDA
            const unsigned int n_blocks =
                (n_entries + block_size - 1) / block_size;
            set_entries<<<n_blocks, block_size>>>(n_entries, 0U,
                                                  column_indices);
            AssertCuda(cudaPeekAtLastError());
            set_entries<<<n_blocks, block_size>>>(n_entries, Number(0), values);
            AssertCuda(cudaPeekAtLastError());
#endif
        } else {
            column_indices = new unsigned int[n_entries];
            values = new Number[n_entries];

#pragma omp parallel for
            for (std::size_t i = 0; i < n_entries; ++i) column_indices[i] = 0;

#pragma omp parallel for
            for (std::size_t i = 0; i < n_entries; ++i) values[i] = 0;
        }
    }

    ~SparseMatrix_SCS() {
        if (memory_space == MemorySpace::CUDA) {
#ifndef DISABLE_CUDA
            cudaFree(chunk_starts);
            cudaFree(chunk_widths);
            cudaFree(column_indices);
            cudaFree(values);
#endif
        } else {
            delete[] chunk_starts;
            delete[] chunk_widths;
            delete[] column_indices;
            delete[] values;
        }
    }

    SparseMatrix_SCS(const SparseMatrix_SCS &other)
        : memory_space(other.memory_space),
          n_chunks(other.n_chunks),
          n_rows(other.n_rows),
          n_entries(other.n_entries) {
        if (memory_space == MemorySpace::CUDA) {
            AssertCuda(cudaMalloc(&chunk_starts,
                                  (n_chunks + 1) * sizeof(std::size_t)));
            AssertCuda(cudaMemcpy(chunk_starts, other.chunk_starts,
                                  (n_chunks + 1) * sizeof(std::size_t),
                                  cudaMemcpyDeviceToDevice));

            AssertCuda(cudaMalloc(&chunk_widths, (n_chunks) * sizeof(std::size_t)));
            AssertCuda(cudaMemcpy(chunk_widths, other.chunk_widths,
                                  (n_chunks) * sizeof(std::size_t),
                                  cudaMemcpyDeviceToDevice));

            AssertCuda(cudaMalloc(&column_indices, n_entries * sizeof(unsigned int)));
            AssertCuda(cudaMemcpy(column_indices, other.column_indices,
                                  n_entries * sizeof(unsigned int),
                                  cudaMemcpyDeviceToDevice));

            AssertCuda(cudaMalloc(&values, n_entries * sizeof(Number)));
            AssertCuda(cudaMemcpy(values, other.values,
                                  n_entries * sizeof(Number),
                                  cudaMemcpyDeviceToDevice));

        } else {
        }
    }

    // do not allow copying matrix
    SparseMatrix_SCS operator=(const SparseMatrix_SCS &other) = delete;

    unsigned int m() const { return n_rows; }

    std::size_t n_nonzero_entries() const { return n_entries; }

    void add_row(unsigned int row, std::vector<unsigned int> &columns_of_row,
                 std::vector<Number> &values_in_row) {
        if (columns_of_row.size() != values_in_row.size()) {
            std::cout << "column_indices and values must have the same size!"
                      << std::endl;
            std::abort();
        }
        const unsigned int chunk = row / C;
        const unsigned int row_in_chunk = row - chunk * C;

        if (columns_of_row.size() > chunk_widths[chunk]) {
            std::cout << "Chunk_width is smaller than the number of values in "
                         "the row."
                      << std::endl;
            std::abort();
        }
        for (unsigned int i = 0; i < columns_of_row.size(); ++i) {
            column_indices[chunk_starts[chunk] + row_in_chunk + i * C] =
                columns_of_row[i];
            values[chunk_starts[chunk] + row_in_chunk + i * C] =
                values_in_row[i];
        }
        // Padding
        for (unsigned int i = columns_of_row.size(); i < chunk_widths[chunk];
             i++) {
            column_indices[chunk_starts[chunk] + row_in_chunk + i * C] =
                columns_of_row[0];
            values[chunk_starts[chunk] + row_in_chunk + i * C] = 0;
        }
    }

    // TODO
    void apply(const Vector<Number> &src, Vector<Number> &dst) const {
        if (m() != src.size_on_this_rank() || m() != dst.size_on_this_rank()) {
            std::cout << "vector sizes of src " << src.size_on_this_rank()
                      << " and dst " << dst.size_on_this_rank()
                      << " do not match matrix size " << m() << std::endl;
            std::abort();
        }

        // main loop for the sparse matrix-vector product
        if (memory_space == MemorySpace::CUDA) {
#ifndef DISABLE_CUDA

            compute_spmv<<<n_chunks, C>>>(n_rows, chunk_starts, chunk_widths,
                                          column_indices, values, src.begin(),
                                          dst.begin());
            AssertCuda(cudaPeekAtLastError());
#endif
        } else { 
            if (C != 4) {
                std::cout << "C != 4 when using cpu. Aborting" << std::endl;
                std::abort();
            }
#pragma omp parallel for
            for (unsigned int chunk = 0; chunk < n_chunks; ++chunk) {
                std::size_t cs = chunk_starts[chunk];
                Number sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                for (std::size_t i = 0; i < chunk_widths[chunk]; ++i){
                        sum0 += values[cs + 0 + i * C] 
                                * src(column_indices[cs + 0 + i * C]);
                        sum1 += values[cs + 1 + i * C] 
                                * src(column_indices[cs + 1 + i * C]);
                        sum2 += values[cs + 2 + i * C] 
                                * src(column_indices[cs + 2 + i * C]);
                        sum3 += values[cs + 3 + i * C] 
                                * src(column_indices[cs + 3 + i * C]);
                }
                dst(chunk * C + 0) = sum0;
                dst(chunk * C + 1) = sum1;
                dst(chunk * C + 2) = sum2;
                dst(chunk * C + 3) = sum3;
            }
        }
    }

    SparseMatrix_SCS copy_to_device() {
        if (memory_space == MemorySpace::CUDA) {
            std::cout << "Copy between device matrices not implemented"
                      << std::endl;
            exit(EXIT_FAILURE);
            // return dummy
            return SparseMatrix_SCS(std::vector<unsigned int>(),
                                    MemorySpace::CUDA, communicator);
        } else {
            std::vector<unsigned int> row_lengths(n_rows);
            for (unsigned int i = 0; i < n_chunks; i++)
                for (unsigned int j = 0; j < C; j++)
                    if (i * C + j < n_rows)
                        row_lengths[i * C + j] = chunk_widths[i];

            SparseMatrix_SCS other(row_lengths, MemorySpace::CUDA,
                                   communicator);
            AssertCuda(cudaMemcpy(other.column_indices, column_indices,
                                  n_entries * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
            AssertCuda(cudaMemcpy(other.values, values,
                                  n_entries * sizeof(Number),
                                  cudaMemcpyHostToDevice));
            return other;
        }
    }

    std::size_t memory_consumption() const {
        return n_entries * (sizeof(Number) + sizeof(unsigned int)) +
               (2 * n_chunks + 1) * sizeof(std::size_t);
    }

   private:
    MPI_Comm communicator;
    std::size_t n_chunks;
    std::size_t n_rows;  // real #rows
    std::size_t *chunk_starts;
    std::size_t *chunk_widths;
    unsigned int *column_indices;
    Number *values;
    std::size_t n_entries;
    MemorySpace memory_space;
};

#endif
