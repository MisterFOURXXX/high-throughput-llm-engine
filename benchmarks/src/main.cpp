#include "load_generator.hpp"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    std::string endpoint = "http://localhost:8000/generate";
    size_t concurrency = 50;
    double rate = 10.0; // requests per second
    size_t duration = 30; // seconds

    if (argc > 1) endpoint = argv[1];
    if (argc > 2) concurrency = std::stoul(argv[2]);
    if (argc > 3) rate = std::stod(argv[3]);
    if (argc > 4) duration = std::stoul(argv[4]);

    SystemLoadGenerator gen(endpoint, concurrency);
    gen.run_poisson_workload(rate, duration);
    gen.export_telemetry_csv("benchmark_results.csv");
    std::cout << "Benchmark completed. Results written to benchmark_results.csv\n";
    return 0;
}