#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

#include "/Users/ivan/IT/HPC-tasks/OpenMP/include/timer.h"
#include "/Users/ivan/IT/HPC-tasks/OpenMP/include/utils.h"
#include "/Users/ivan/IT/HPC-tasks/OpenMP/include/utils_matrix.h"
#include "/Users/ivan/IT/HPC-tasks/OpenMP/include/utils_matrix_special.h"
#include "/Users/ivan/IT/HPC-tasks/OpenMP/include/tasks/task5_special_matrices.h"

#include <nlohmann/json.hpp>
#include <omp.h>

using json = nlohmann::json;

int main() {
    std::ifstream fin("config.json");
    if (!fin.is_open()) {
        std::cerr << "❌ Ошибка: не найден config.json\n";
        return 1;
    }

    json cfg;
    fin >> cfg;

    std::vector<size_t> sizes      = cfg.value("sizes",      std::vector<size_t>{500, 1000});
    std::vector<int>    threads    = cfg.value("threads",    std::vector<int>{1, 2, 4, 8});
    int repeats                   = cfg.value("repeats",    3);
    std::string output_file       = cfg.value("output",     "results/task5_results.csv");

    std::string matrix_type       = cfg.value("matrix_type", "banded");
    size_t bandwidth              = cfg.value("bandwidth",   3);

    // ------------------ запуск тестов ------------------
    for (int t : threads) {
        omp_set_num_threads(t);
        std::cout << "\n🚀 Потоки: " << t << "\n";

        for (size_t N : sizes) {

            double best_time = 1e9;
            double result_val = 0.0;

            for (int r = 0; r < repeats; r++) {
                Timer timer;
                timer.start();

                // ----------- генерация специальной матрицы --------------
                std::vector<double> A;

                if (matrix_type == "banded") {
                    A = generate_banded_matrix(N, bandwidth);
                }
                else if (matrix_type == "lower") {
                    A = generate_lower_triangular(N);
                }
                else if (matrix_type == "upper") {
                    A = generate_upper_triangular(N);
                }
                else {
                    std::cerr << "❌ Неизвестный matrix_type: " << matrix_type << "\n";
                    return 1;
                }

                result_val = run_task5_special_matrices(A, N, t);

                double elapsed = timer.stop();
                best_time = std::min(best_time, elapsed);
            }

            save_result_csv(output_file, 
                            "task5_special_matrices",
                            t, 
                            N, 
                            best_time, 
                            result_val);

            std::cout << "[OK] N=" << N
                      << " time=" << best_time
                      << " result=" << result_val << "\n";
        }
    }

    std::cout << "\n📁 Результаты сохранены в " << output_file << "\n";
    return 0;
}