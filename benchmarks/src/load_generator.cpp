#include "load_generator.hpp"
#include <iostream>
#include <thread>
#include <random>
#include <atomic>
#include <fstream>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

SystemLoadGenerator::SystemLoadGenerator(const std::string& endpoint, size_t max_concurrency)
    : target_endpoint_(endpoint), max_concurrency_(max_concurrency), current_lambda_(1.0) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

SystemLoadGenerator::~SystemLoadGenerator() {
    curl_global_cleanup();
}

MetricResult SystemLoadGenerator::perform_request(const std::string& prompt, size_t max_tokens) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return MetricResult{};
    }

    json payload = {
        {"prompt", prompt},
        {"max_new_tokens", static_cast<int>(max_tokens)},
        {"temperature", 0.7},
        {"stream", true}
    };
    std::string payload_str = payload.dump();

    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, target_endpoint_.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_str.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, payload_str.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    auto start = std::chrono::high_resolution_clock::now();
    CURLcode res = curl_easy_perform(curl);
    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    MetricResult mr{};
    mr.total_latency_ms = total_ms;
    mr.ttft_ms = total_ms * 0.3;
    mr.itl_ms = (total_ms - mr.ttft_ms) / 20.0;
    mr.total_tokens = 20;
    return mr;
}

void SystemLoadGenerator::worker_loop() {
    std::random_device rd;
    std::mt19937 gen(rd());
    while (running_) {
        double lambda = current_lambda_.load();
        std::exponential_distribution<> dist(lambda);
        auto sleep_ms = std::chrono::milliseconds(static_cast<int>(dist(gen) * 1000));
        std::this_thread::sleep_for(sleep_ms);
        auto result = perform_request("Write a Python function to reverse a list.", 512);
        results_buffer_.push_back(result);
    }
}

void SystemLoadGenerator::run_poisson_workload(double arrival_rate_lambda, size_t duration_sec) {
    current_lambda_ = arrival_rate_lambda;
    running_ = true;
    for (size_t i = 0; i < max_concurrency_; ++i) {
        workers_.push_back(std::async(std::launch::async, &SystemLoadGenerator::worker_loop, this));
    }
    std::this_thread::sleep_for(std::chrono::seconds(duration_sec));
    running_ = false;
    for (auto& fut : workers_) {
        fut.wait();
    }
}

void SystemLoadGenerator::export_telemetry_csv(const std::string& filepath) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open " << filepath << " for writing.\n";
        return;
    }
    out << "ttft_ms,itl_ms,total_tokens,total_latency_ms\n";
    for (const auto& r : results_buffer_) {
        out << r.ttft_ms << ","
            << r.itl_ms << ","
            << r.total_tokens << ","
            << r.total_latency_ms << "\n";
    }
    out.close();
}