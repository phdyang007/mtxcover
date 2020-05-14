#pragma once

#include "common.h"

enum class ImplVersion { ORIGINAL_CPU, ORIGINAL_GPU, ORIGINAL_GPU_MG };

MeasureTimer Invoke(const ImplVersion version, bool print_result,
                    DataSet *dataset, int &backtrace_num, int &total_conflict_num);
MeasureTimer Invoke(const ImplVersion version, bool print_result,
                    DataSets *datasets, int &backtrace_num, int &total_conflict_num);
