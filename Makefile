.PHONY: all build-benchmark run-benchmark clean

all: build-benchmark

build-benchmark:
	mkdir -p benchmarks/build
	cd benchmarks/build && cmake .. && make

run-benchmark: build-benchmark
	./benchmarks/build/loadgen

clean:
	rm -rf benchmarks/build
	rm -f benchmark_results.csv
	rm -rf artifacts/*