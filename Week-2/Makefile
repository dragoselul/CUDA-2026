target  = reduction
# Automatically finds all .cu and .cpp files in the current directory
CU_SRCS  = $(wildcard *.cu)
CPP_SRCS = $(wildcard *.cpp)

# Converts the source list into a list of .o files
objs     = $(CU_SRCS:.cu=.o) $(CPP_SRCS:.cpp=.o)

# Flags
opt      = -g -O3
para     = -fopenmp
# Architecture for Hopper (sm_90), change if using A100 (sm_80) or RTX 30 (sm_86)
arch     = sm_89

# Compilers
cxx      = nvc++ -cuda -mp=gpu -minfo --diag_suppress integer_sign_change,cuda_compile
cxxflags = $(opt) $(para)

nvcc     = nvcc
cuflags = --compiler-options "$(opt) $(para)" --ptxas-options=-v -arch=$(arch) -lineinfo -diag-suppress 68

# --- Build Rules ---

all: $(target)

# Linking: Uses nvc++ to bring everything together
$(target): $(objs) 
	$(cxx) $(cxxflags) -o $@ $(objs)

# Rule for CUDA files
%.o: %.cu
	$(nvcc) $(cuflags) -c $< -o $@

# Rule for C++ files
%.o: %.cpp
	$(cxx) $(cxxflags) -c $< -o $@

# Cleanup
clean:
	rm -f *.o $(target)

.PHONY: all clean
