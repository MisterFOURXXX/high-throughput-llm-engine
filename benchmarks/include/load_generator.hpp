#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <future>
#include <memory>
#include <atomic>
#include <curl/curl.h>

struct RequestPayload {
    std::string prompt;
    size_t input_tokens;
    size_t max_output_tokens;
    std::chrono::high_resolution_clock::time_point dispatch_time;
};

struct MetricResult {
    double ttft_ms;
    double itl_ms;
    size_t total_tokens;
    double total_latency_ms;
};

class SystemLoadGenerator {
public:
    explicit SystemLoadGenerator(const std::string& endpoint, size_t max_concurrency);
    ~SystemLoadGenerator();

    void run_poisson_workload(double arrival_rate_lambda, size_t duration_sec);
    void export_telemetry_csv(const std::string& filepath);

private:
    void worker_loop();
    MetricResult perform_request(const std::string& prompt, size_t max_tokens);

    std::string target_endpoint_;
    size_t max_concurrency_;
    std::vector<MetricResult> results_buffer_;
    std::vector<std::future<void>> workers_;
    bool running_ = false;
    std::atomic<double> current_lambda_;
};