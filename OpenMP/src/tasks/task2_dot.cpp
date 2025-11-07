#include <vector>
#include <omp.h>

double run_task2_dot(const std::vector<double>& a,
                     const std::vector<double>& b,
                     int threads)
{
    double result = 0.0;
    size_t n = a.size();

    #pragma omp parallel for num_threads(threads) reduction(+:result)
    for (size_t i = 0; i < n; ++i)
        result += a[i] * b[i];

    return result;
}