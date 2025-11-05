CXX = g++-15
CXXFLAGS = -O3 -fopenmp -Iinclude -I/opt/homebrew/include -I/usr/local/include
LDFLAGS = -fopenmp


SRC = $(wildcard src/*.cpp src/tasks/*.cpp)
OBJ = $(SRC:.cpp=.o)
TARGET = openmp_lab
RESULTS_DIR = results

$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

all: $(RESULTS_DIR) $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

run: all
	./$(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)
