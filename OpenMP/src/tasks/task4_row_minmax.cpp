#include "/Users/ivan/IT/HPC-tasks/OpenMP/include/tasks/task4_row_minmax.h"
#include <omp.h>
#include <limits>
#include <algorithm>

double run_task4_row_minmax(const std::vector<double>& A, size_t N, int threads) {
    omp_set_num_threads(threads);

    auto getA = [&](size_t i, size_t j) {
        return A[i * N + j];
    };

    double global_max = -std::numeric_limits<double>::infinity();

    #pragma omp parallel
    {
        double local_max = -std::numeric_limits<double>::infinity();

        #pragma omp for nowait
        for (size_t i = 0; i < N; ++i) {
            double row_min = std::numeric_limits<double>::infinity();

            for (size_t j = 0; j < N; ++j) {
                double v = getA(i, j);
                if (v < row_min)
                    row_min = v;
            }

            if (row_min > local_max)
                local_max = row_min;
        }

        #pragma omp critical
        {
            if (local_max > global_max)
                global_max = local_max;
        }
    }

    return global_max;
}