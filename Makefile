# Makefile for Image Processor with NVIDIA NPP and OpenCV

# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall

# Directories
SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.cpp)

# Object files
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Output binary
TARGET := $(BIN_DIR)/image_processor.exe

# Libraries
OPENCV_LIBS := `pkg-config --cflags --libs opencv4`
NPP_LIBS := -lnppicc -lnppisu -lnppist -lnpps -lnppif -lcudart

# Default target
all: $(TARGET)

# Create directories if not existing
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build target
$(TARGET): $(BUILD_DIR) $(BIN_DIR) $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(OPENCV_LIBS) $(NPP_LIBS)

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(OPENCV_LIBS)

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Phony targets
.PHONY: all clean

