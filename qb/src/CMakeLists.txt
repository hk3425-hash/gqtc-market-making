cmake_minimum_required(VERSION 3.15)
project(TrackerExtension LANGUAGES CXX)

find_package(pybind11 REQUIRED)

pybind11_add_module(tracker tracker.cpp)

# Explicitly force C++17 on the pybind target
target_compile_features(tracker PRIVATE cxx_std_17)

# Optimize for speed (bypasses the need to specify Release mode manually)
target_compile_options(tracker PRIVATE -Wall -O3)