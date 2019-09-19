#pragma once

#include <iostream>

#include "common.h"

std::vector<DataSet> ReadDataSet(const std::string &cofig_file);

DataSets CombineDatasets(const std::vector<DataSet>& dataset);
