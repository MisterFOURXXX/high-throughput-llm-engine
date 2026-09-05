#include "metrics_collector.hpp"

void MetricsCollector::write_csv(const std::string& path,
                                 const std::vector<MetricResult>& results) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open " << path << " for writing.\n";
        return;
    }
    out << "ttft_ms,itl_ms,total_tokens,total_latency_ms\n";
    for (const auto& r : results) {
        out << r.ttft_ms << ","
            << r.itl_ms << ","
            << r.total_tokens << ","
            << r.total_latency_ms << "\n";
    }
    out.close();
    std::cout << "Metrics written to " << path << "\n";
}