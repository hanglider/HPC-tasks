#pragma once
#include <vector>
#include <random>

inline std::vector<double> generate_matrix(size_t N) {
    std::vector<double> A(N * N);
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

    for (size_t i = 0; i < N * N; ++i)
        A[i] = dist(rng);

    return A;
}