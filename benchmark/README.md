# How To Use the benchmark

1. Prepare the data
  - Prepare the matrix file and the col name file, save them separately
2. Write a test case config file, Here are the format
```
N_TEST_CASE
N_VERTEX_IN_TEST_CASE_1
N_ROW_IN_TEST_CASE_1 N_COL_IN_TEST_CASE_1
FILE_PATH_TO_THE_MATRIX_FILE_IN_TEST_CASE_1
FILE_PATH_TO_THE_COL_FILE_IN_TEST_CASE_1

...

N_VERTEX_IN_TEST_CASE_N
N_ROW_IN_TEST_CASE_N N_COL_IN_TEST_CASE_N
FILE_PATH_TO_THE_MATRIX_FILE_IN_TEST_CASE_N
FILE_PATH_TO_THE_COL_FILE_IN_TEST_CASE_N
```

3. Use the config. Add the test_config file to the `test_configs` vector in `benchmark/benchmark.cc`
4. `make`
5. `./build/benchmark.bin`
