#pragma once

#include <iostream>

#include "common.h"

std::vector<DataSet>
ReadDataSetFromMatrixFolder(const std::string &matrix_folder,
                            const std::string &result_folder);

DataSets CombineDatasets(const std::vector<DataSet> &dataset);
