#include "task3_integral.h"
#include <omp.h>

static inline double f(double x) { return 4.0 / (1.0 + x*x); }

double run_task3_integral(const std::vector<double>& dummy, int threads) {
    const int N = static_cast<int>(dummy.size()); 
    const double a = 0.0, b = 1.0;
    const double h = (b - a) / N;

    double sum = 0.0;
    omp_set_num_threads(threads);
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        double x = a + i * h;     
        sum += f(x);
    }
    return h * sum;               
}