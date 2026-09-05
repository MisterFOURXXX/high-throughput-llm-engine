cmake_minimum_required(VERSION 3.15)

find_package(CURL REQUIRED)
find_package(Threads REQUIRED)

add_executable(loadgen
    src/main.cpp
    src/load_generator.cpp
    src/metrics_collector.cpp
)

target_include_directories(loadgen PRIVATE include)
target_link_libraries(loadgen CURL::libcurl Threads::Threads)