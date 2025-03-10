# Compiler
MPICXX = mpicxx

# Flags
CXXFLAGS = -O2 -Wall
LDFLAGS = -lz

# Source files
SRC = scatter_test.cpp

# Output executable
TARGET = scatter_test

# Rules
all: $(TARGET)

$(TARGET): $(SRC)
	$(MPICXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)