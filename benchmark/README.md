# How To Use the benchmark

1. Prepare the data
  - Can look at the example in `matrix` folder.
  - The `n_data.txt` is the number of matrix
  - The `col_cnt.txt` is the number of col
  - The `row_cnt.txt` is the number of row
  - The `col.txt` is the column name group
  - The `vertex.txt` is the number of vertex
  - The `matrix.txt` is the dl matrix
2. `make`
3. `./build/benchmark.bin`
4. Can select another testing dataset by adding to the `test_configs` in the `benchmark.cc`
