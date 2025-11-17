#include "/Users/ivan/IT/HPC-tasks/OpenMP/include/utils.h"

std::vector<double> generate_vector(size_t n, double min, double max) {
    std::mt19937 gen(42); // фиксируем seed для воспроизводимости
    std::uniform_real_distribution<double> dist(min, max);
    std::vector<double> v(n);
    for (auto& x : v) x = dist(gen);
    return v;
}

void save_result_csv(const std::string& filename, const std::string& task_name,
                     int threads, size_t size, double time, double result) {
    std::ofstream fout(filename, std::ios::app);
    if (fout.tellp() == 0)
        fout << "task,threads,size,time,result\n";

    fout << task_name << "," << threads << "," << size << "," << time << "," << result << "\n";

    std::cout << "🚀 [" << threads
              << (threads == 1 ? " поток" : " потоков") << "] "
              << task_name
              << ", N=" << size
              << ", ⏱ " << time << "s"
              << ", 🧮 result=" << result
              << std::endl;
}
