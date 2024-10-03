# Define the compiler and flags
CXX = g++
CXXFLAGS = -Wall -g

# Define the target executable
TARGET = my_program

# Define the source files and object files
SRC = main.cpp functions.cpp funcs.h
OBJ = $(SRC:.cpp=.o)

# Rule to build the final executable
$(TARGET):$(OBJ)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up the build (remove object files and executable)
clean:
	rm -f $(OBJ) $(TARGET)