#include "/Users/ivan/IT/HPC-tasks/OpenMP/include/tasks/task6_schedule.h"
#include <omp.h>
#include <cmath>
#include <string>
#include <limits>
#include <iostream>

static inline double heavy_work(size_t i) {
    double x = 0;
    for (int k = 0; k < 5000; k++) {
        x += std::sin(i * 0.00001 + k);
    }
    return x;
}

double run_task6_schedule(size_t N, int threads, const std::string& mode)
{
    omp_set_num_threads(threads);

    double global_sum = 0.0;

    auto t0 = omp_get_wtime();

    if (mode == "static") {
#pragma omp parallel for schedule(static) reduction(+:global_sum)
        for (size_t i = 0; i < N; i++)
            global_sum += heavy_work(i);
    }

    else if (mode == "dynamic") {
#pragma omp parallel for schedule(dynamic) reduction(+:global_sum)
        for (size_t i = 0; i < N; i++)
            global_sum += heavy_work(i);
    }

    else if (mode == "guided") {
#pragma omp parallel for schedule(guided) reduction(+:global_sum)
        for (size_t i = 0; i < N; i++)
            global_sum += heavy_work(i);
    }

    else {
        std::cerr << "❌ Unknown schedule mode: " << mode << "\n";
        return -1;
    }

    double t1 = omp_get_wtime();

    return t1 - t0;
}
