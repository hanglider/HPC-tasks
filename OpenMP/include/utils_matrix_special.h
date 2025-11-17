#pragma once
#include <vector>
#include <random>

// Простейший ГПСЧ
inline double rnd() {
    static thread_local std::mt19937_64 rng(123456);
    static thread_local std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    return dist(rng);
}

// ЛЕНТОЧНАЯ МАТРИЦА (bandwidth = k)
inline std::vector<double> generate_banded_matrix(size_t N, size_t k) {
    std::vector<double> A(N * N, 0.0);

    for (size_t i = 0; i < N; ++i) {
        size_t j_start = (i > k ? i - k : 0);
        size_t j_end   = std::min(N, i + k + 1);
        for (size_t j = j_start; j < j_end; ++j)
            A[i * N + j] = rnd();
    }
    return A;
}

// НИЖНЕТРЕУГОЛЬНАЯ
inline std::vector<double> generate_lower_triangular(size_t N) {
    std::vector<double> A(N * N, 0.0);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j <= i; ++j)
            A[i * N + j] = rnd();
    return A;
}

// ВЕРХНЕТРЕУГОЛЬНАЯ
inline std::vector<double> generate_upper_triangular(size_t N) {
    std::vector<double> A(N * N, 0.0);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = i; j < N; ++j)
            A[i * N + j] = rnd();
    return A;
}