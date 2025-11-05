#include "task1_minmax.h"
#include <omp.h>
#include <algorithm>

double run_task1_minmax(const std::vector<double>& data, int threads, bool find_min) {
    omp_set_num_threads(threads);
    double result = data[0];

    #pragma omp parallel for reduction(min:result) if(find_min)
    for (size_t i = 0; i < data.size(); ++i) {
        if (find_min)
            result = std::min(result, data[i]);
    }

    #pragma omp parallel for reduction(max:result) if(!find_min)
    for (size_t i = 0; i < data.size(); ++i) {
        if (!find_min)
            result = std::max(result, data[i]);
    }

    return result;
}