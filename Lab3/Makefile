# ============================================================
# Lab3 - Correlation Assignment Makefile
# UCS645: Parallel & Distributed Computing
# ============================================================

CXX      = g++
CXXFLAGS = -std=c++11 -Wall -O2 -fopenmp -march=native

TARGET  = correlation
SOURCES = main.cpp correlate.cpp
OBJECTS = $(SOURCES:.cpp=.o)
HEADERS = correlate.h

# Default: build the executable
all: $(TARGET)

# Link
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Compile each .cpp -> .o
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Quick run with small matrix
run: $(TARGET)
	./$(TARGET) 100 500

# perf stat — sequential (1 thread)
# Usage: make perf_seq NY=500 NX=2000
perf_seq: $(TARGET)
	perf stat ./$(TARGET) $(NY) $(NX) 1

# perf stat — parallel (T threads)
# Usage: make perf_par NY=500 NX=2000 T=4
perf_par: $(TARGET)
	perf stat ./$(TARGET) $(NY) $(NX) $(T)

# Remove build files
clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all run clean perf_seq perf_par
