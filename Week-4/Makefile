target  = reduction
CU_SRCS  = $(wildcard *.cu)
CPP_SRCS = $(wildcard *.cpp)

objs     = $(CU_SRCS:.cu=.o) $(CPP_SRCS:.cpp=.o)

# Flags
opt      = -g -O3
para     = -fopenmp
# GH200 is Hopper Architecture
arch     = sm_90

# Compilers
# Added -tp=neoverse-v2 to optimize the C++ code for the Grace CPU cores
cxx = nvc++ -cuda -mp=gpu -tp=neoverse-v2 -Minfo=accel,inline --diag_suppress integer_sign_change,cuda_compile
cxxflags = $(opt) $(para)

nvcc     = nvcc
# Using -Xcompiler to pass optimization flags to the host compiler
cuflags  = -arch=$(arch) -Xcompiler "$(opt) $(para)" -Xptxas -v -lineinfo -diag-suppress 68

# --- Build Rules ---

all: $(target)

$(target): $(objs) 
	$(cxx) $(cxxflags) -o $@ $(objs)

%.o: %.cu
	$(nvcc) $(cuflags) -c $< -o $@

%.o: %.cpp
	$(cxx) $(cxxflags) -c $< -o $@

clean:
	rm -f *.o $(target)

.PHONY: all clean