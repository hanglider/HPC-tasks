#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "/Users/ivan/IT/HPC-tasks/OpenMP/include/timer.h"
#include "utils.h"
#include "tasks/task1_minmax.h"
#include <nlohmann/json.hpp>

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
    std::vector<size_t> sizes = cfg.value("sizes", std::vector<size_t>{1000, 100000, 10000000});
    std::vector<int> threads_list = cfg.value("threads", std::vector<int>{1, 2, 4, 8});
    int repeats = cfg.value("repeats", 3);
    bool find_min = cfg.value("find_min", true);
    std::string output_file = cfg.value("output", "results/task1_minmax.csv");

    for (auto threads : threads_list) {
        omp_set_num_threads(threads);
        std::cout << "\n🚀 Тестирование с " << threads << (threads == 1 ? " потоком" : " потоками") << ":\n";

        for (auto size : sizes) {
            auto data = generate_vector(size);
            double best_time = 1e9;
            double result_val = 0.0;

            for (int r = 0; r < repeats; ++r) {
                Timer t;
                t.start();
                result_val = run_task1_minmax(data, threads, find_min);
                double elapsed = t.stop();
                best_time = std::min(best_time, elapsed);
            }

            save_result_csv(output_file, task_name, threads, size, best_time, result_val);
        }
    }


    std::cout << "Результаты сохранены в " << output_file << std::endl;
    return 0;
}
