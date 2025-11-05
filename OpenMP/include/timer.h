#pragma once
#include <omp.h>

class Timer {
    double start_time;
public:
    void start() { start_time = omp_get_wtime(); }
    double stop() { return omp_get_wtime() - start_time; }
};