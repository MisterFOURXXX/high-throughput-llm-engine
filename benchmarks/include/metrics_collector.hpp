#pragma once

#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>
#include "load_generator.hpp"   // <-- ADD THIS (defines MetricResult)

class MetricsCollector {
public:
    static void write_csv(const std::string& path,
                          const std::vector<MetricResult>& results);
};