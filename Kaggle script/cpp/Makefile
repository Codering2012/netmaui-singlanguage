# C++ ONNX Runtime Streaming Inference Engine Makefile
CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra

TARGET = inference_engine
SRCS = inference_engine.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -I. $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o *.exe

.PHONY: all clean
