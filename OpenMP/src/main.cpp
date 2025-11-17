#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

#include "/Users/ivan/IT/HPC-tasks/OpenMP/include/timer.h"
#include "/Users/ivan/IT/HPC-tasks/OpenMP/include/utils.h"
#include "/Users/ivan/IT/HPC-tasks/OpenMP/include/utils_matrix.h"
#include "/Users/ivan/IT/HPC-tasks/OpenMP/include/tasks/task4_row_minmax.h"

#include <nlohmann/json.hpp>
#include <omp.h>

using json = nlohmann::json;

int main() {
    std::ifstream fin("config.json");
    if (!fin.is_open()) {
        std::cerr << "Ошибка: не найден config.json\n";
        return 1;
    }

    json cfg;
    fin >> cfg;

    std::string task_name = cfg.value("task", "task1_minmax");
    std::vector<size_t> sizes = cfg.value("sizes", std::vector<size_t>{1000});
    std::vector<int> threads_list = cfg.value("threads", std::vector<int>{1, 2, 4, 8});
    int repeats = cfg.value("repeats", 3);
    bool find_min = cfg.value("find_min", true);
    std::string output_file = cfg.value("output", "results/out.csv");

    for (int threads : threads_list) {
        omp_set_num_threads(threads);
        std::cout << "\n🚀 Потоки: " << threads << "\n";

        for (size_t N : sizes) {
            double best_time = 1e9;
            double result_val = 0.0;

            for (int r = 0; r < repeats; ++r) {
                Timer t;
                t.start();
                auto A = generate_matrix(N);
                result_val = run_task4_row_minmax(A, N, threads);
                double elapsed = t.stop();
                best_time = std::min(best_time, elapsed);
            }

            save_result_csv(output_file, task_name, threads, N, best_time, result_val);

            std::cout << "[OK] N=" << N
                      << " time=" << best_time
                      << " result=" << result_val << "\n";
        }
    }

    std::cout << "\nРезультаты сохранены в " << output_file << "\n";
    return 0;
}