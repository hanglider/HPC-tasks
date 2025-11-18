#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

#include "/Users/ivan/IT/HPC-tasks/OpenMP/include/timer.h"
#include "/Users/ivan/IT/HPC-tasks/OpenMP/include/utils.h"
#include "/Users/ivan/IT/HPC-tasks/OpenMP/include/tasks/task6_schedule.h"

#include <nlohmann/json.hpp>
#include <omp.h>

using json = nlohmann::json;

int main() {
    std::ifstream fin("config.json");
    if (!fin.is_open()) {
        std::cerr << "❌ Не найден config.json\n";
        return 1;
    }

    json cfg;
    fin >> cfg;

    std::vector<size_t> sizes   = cfg.value("sizes",   std::vector<size_t>{200000});
    std::vector<int> threads    = cfg.value("threads", std::vector<int>{1,2,4,8});
    std::vector<std::string> modes = 
        cfg.value("schedule_mode", std::vector<std::string>{"static", "dynamic", "guided"});
    int repeats = cfg.value("repeats", 3);

    std::string output_file = cfg.value("output", "results/task6_schedule.csv");

    // ---- запуск тестов ----
    for (int t : threads) {
        omp_set_num_threads(t);
        std::cout << "\n🚀 Потоки: " << t << "\n";

        for (size_t N : sizes) {

            for (auto& mode : modes) {

                double best_time = 1e9;
                double dummy = 0.0;

                for (int r = 0; r < repeats; r++) {
                    Timer timer;
                    timer.start();

                    dummy = run_task6_schedule(N, t, mode);

                    double elapsed = timer.stop();
                    best_time = std::min(best_time, elapsed);
                }

                save_result_csv(output_file,
                                "task6_schedule_" + mode,
                                t,
                                N,
                                best_time,
                                dummy);

                std::cout << "[OK] mode=" << mode
                          << " N=" << N
                          << " time=" << best_time << "\n";
            }
        }
    }

    std::cout << "\n📁 Результаты сохранены в " << output_file << "\n";
    return 0;
}