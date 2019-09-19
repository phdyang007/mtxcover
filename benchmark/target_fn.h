#pragma once

#include "common.h"

enum class ImplVersion { ORIGINAL_CPU, ORIGINAL_GPU, ORIGINAL_GPU_MG };

MeasureTimer Invoke(const ImplVersion version, bool print_result,
                    DataSet *dataset);
MeasureTimer Invoke(const ImplVersion version, bool print_result,
                    DataSets *datasets);
