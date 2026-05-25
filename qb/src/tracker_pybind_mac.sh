g++-15 -O3 -Wall -shared -std=c++20 -fPIC \
    $(python3 -m pybind11 --includes) \
    -undefined dynamic_lookup \
    src/tracker.cpp \
    -o tracker$(python3-config --extension-suffix)
