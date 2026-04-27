# ============================================================
# Makefile — CUDA DIY Exercises 1-5 (Customized for MX130)
# ============================================================
# Usage:
#   make all        — build all 5 exercises into build/
#   make clean      — remove the build/ directory
#   make run_ex01   — build and run Exercise 1
# ============================================================

SM        := sm_50
NVCC      := nvcc
NVCCFLAGS := -O2 -arch=$(SM) -lineinfo
BUILD_DIR := build

# Libraries required for different exercises
LIBS_BASE  := -lm
LIBS_BLAS  := -lcublas $(LIBS_BASE)
LIBS_FULL  := -lcudnn -lcublas $(LIBS_BASE)

# Dynamically mapped targets
TARGETS = $(BUILD_DIR)/ex01_cuda_basics \
          $(BUILD_DIR)/ex02_memory_hierarchy \
          $(BUILD_DIR)/ex03_ml_primitives \
          $(BUILD_DIR)/ex04_cnn_layers \
          $(BUILD_DIR)/ex05_mnist_cnn

.PHONY: all clean run_ex01 run_ex02 run_ex03 run_ex04 run_ex05

all: $(TARGETS)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Individual build rules to link specific libraries per exercise
$(BUILD_DIR)/ex01_cuda_basics: ex01_cuda_basics.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BASE)
	@echo "[✓] Built $@"

$(BUILD_DIR)/ex02_memory_hierarchy: ex02_memory_hierarchy.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BASE)
	@echo "[✓] Built $@"

$(BUILD_DIR)/ex03_ml_primitives: ex03_ml_primitives.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BASE)
	@echo "[✓] Built $@"

$(BUILD_DIR)/ex04_cnn_layers: ex04_cnn_layers.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BLAS)
	@echo "[✓] Built $@"

$(BUILD_DIR)/ex05_mnist_cnn: ex05_mnist_cnn.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_FULL)
	@echo "[✓] Built $@"

# Convenience rules to run them directly
run_ex01: $(BUILD_DIR)/ex01_cuda_basics
	./$(BUILD_DIR)/ex01_cuda_basics

run_ex02: $(BUILD_DIR)/ex02_memory_hierarchy
	./$(BUILD_DIR)/ex02_memory_hierarchy

run_ex03: $(BUILD_DIR)/ex03_ml_primitives
	./$(BUILD_DIR)/ex03_ml_primitives

run_ex04: $(BUILD_DIR)/ex04_cnn_layers
	./$(BUILD_DIR)/ex04_cnn_layers

run_ex05: $(BUILD_DIR)/ex05_mnist_cnn
	./$(BUILD_DIR)/ex05_mnist_cnn

clean:
	rm -rf $(BUILD_DIR)
	@echo "[✓] Cleaned"
