# Compiler
MPICXX = mpic++

# Flags
CXXFLAGS = -O2 -Wall
LDFLAGS = -lz

# Source files
SRC = MPI_random_walk.cpp functions.cpp

# Output executable
TARGET = MPI_random_walk

# Rules
all: $(TARGET)

$(TARGET): $(SRC)
	$(MPICXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)