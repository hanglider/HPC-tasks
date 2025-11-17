#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <iostream>

std::vector<double> generate_vector(size_t n, double min = -100.0, double max = 100.0);
void save_result_csv(const std::string& filename, const std::string& task_name,
                     int threads, size_t size, double time, double result);