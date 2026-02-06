# *********************************************************************
# Learning-CUDA Makefile
# Targets:
#   make                        : Build + run tests (default, non-verbose)
#   make build                  : Only compile (no run)
#   make run                    : Run tests (after build, non-verbose)
#   make run VERBOSE=true       : Run tests with verbose output
#   make clean                  : Delete temporary files
# *********************************************************************

# -------------------------------
# Configuration
# -------------------------------
PLATFORM        ?= nvidia
PLATFORM_DEFINE ?= -DPLATFORM_NVIDIA
STUDENT_SUFFIX  := cu
CFLAGS          := -std=c++17 -O0
EXTRA_LIBS      := 

# Compiler & Tester object selection based on PLATFORM
ifeq ($(PLATFORM),nvidia)
    CC                  := nvcc
    TEST_OBJ            := tester/tester_nv.o
	PLATFORM_DEFINE := -DPLATFORM_NVIDIA
# CUDA_HOME and CUDA_LIBDIR are only used for nvidia platform
	CUDA_HOME ?= /usr/local/cuda
	CUDA_LIBDIR ?= $(CUDA_HOME)/lib64
	EXTRA_LIBS := -L$(CUDA_HOME)/lib64 -lcudart -Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64
else ifeq ($(PLATFORM),iluvatar)
	COREX_ROOT ?= /usr/local/corex
	CC := $(COREX_ROOT)/bin/clang++
	TEST_OBJ := tester/tester_iluvatar.o
	PLATFORM_DEFINE := -DPLATFORM_ILUVATAR
	INC := -I$(COREX_ROOT)/include

# KEY: to make clang++ generate device core accoring to ivcore
	# fat device image
	ILUVATAR_ARCHES ?= ivcore10 ivcore11 
	IVCORE_ARCH_FLAGS := $(foreach a,$(ILUVATAR_ARCHES),--cuda-gpu-arch=$(a))
	# compile .cu
	IVCORE_CFLAGS := -std=c++17 -O3 -fPIC \
	        $(IVCORE_ARCH_FLAGS) \
	        --cuda-path=$(COREX_ROOT) \
	        $(INC) $(PLATFORM_DEFINE)
	IVCORE_LDFLAGS := --cuda-path=$(COREX_ROOT)
# link configuration
	EXTRA_LIBS := -L$(COREX_ROOT)/lib64 -lcudart -Wl,-rpath,$(COREX_ROOT)/lib64
else ifeq ($(PLATFORM),moore)
	MUSA_ROOT       ?= /usr/local/musa
    CC              := $(MUSA_ROOT)/bin/mcc
	CFLAGS          := -std=c++11 -O3
    TEST_OBJ        := tester/tester_moore.o
	STUDENT_SUFFIX  := mu
	PLATFORM_DEFINE := -DPLATFORM_MOORE
# include .h (musa and stdlib)（
	INC             := -I$(MUSA_ROOT)/include \
	                   -I/usr/include/c++/11 \
	                   -I/usr/include/x86_64-linux-gnu/c++/11
# link configuration
# - MUSA runtime lib (musart)
# - GCC 11 std path
# - rpath : make sure: find the ddl
	EXTRA_LIBS      := $(INC) \
	                   -L$(MUSA_ROOT)/lib \
	                   -lmusart \
	                   -L/usr/lib/gcc/x86_64-linux-gnu/11/ \
	                   -Wl,-rpath,$(MUSA_ROOT)/lib


else ifeq ($(PLATFORM),metax)
    MACA_ROOT       ?= /opt/maca
    CC              := mxcc
    CFLAGS          := -std=c++17 -O3
    TEST_OBJ        := tester/tester_metax.o
    STUDENT_SUFFIX  := maca
    PLATFORM_DEFINE := -DPLATFORM_METAX
# link configuration
    EXTRA_LIBS      := -I$(MACA_ROOT)/include \
                       -L$(MACA_ROOT)/lib \
                       -lmcruntime \
                       -Wl,-rpath,$(MACA_ROOT)/lib
else
    $(error Unsupported PLATFORM '$(PLATFORM)' (expected: nvidia, iluvatar, moore, metax))
endif

# Executable name
TARGET                  := test_kernels
# Kernel implementation
STUDENT_SRC             := src/kernels.$(STUDENT_SUFFIX) 
# Compiled student object (auto-generated)
STUDENT_OBJ             := $(addsuffix .o,$(basename $(STUDENT_SRC)))
# Tester's actual verbose argument (e.g., --verbose, -v)
TEST_VERBOSE_FLAG       := --verbose
# User-provided verbose mode (true/false; default: false)
VERBOSE                 :=  

# -------------------------------
# Process User Input (VERBOSE → Tester Flag)
# -------------------------------
# Translates `VERBOSE=true` (case-insensitive) to the tester's verbose flag.
# If VERBOSE is not "true" (or empty), no flag is passed.
VERBOSE_ARG := $(if $(filter true True TRUE, $(VERBOSE)), $(TEST_VERBOSE_FLAG),)

# -------------------------------
# Phony Targets
# -------------------------------
.PHONY: all build run clean

# Default target: Build + run tests (non-verbose)
all: build run

# Build target: Compile student code + link with test logic
build: $(TARGET)

# Run target: Execute tests (supports `VERBOSE=true` for verbose output)
run: $(TARGET)
	@echo "=== Running tests (output from $(STUDENT_OBJ)) ==="
	@# Show verbose mode status (friendly for users)
	@if [ -n "$(VERBOSE_ARG)" ]; then \
	    echo "=== Verbose mode: Enabled (using '$(TEST_VERBOSE_FLAG)') ==="; \
	else \
	    echo "=== Verbose mode: Disabled ==="; \
	fi
	./$(TARGET) $(VERBOSE_ARG)

# Clean target: Delete temporary files (executable + src object)
clean:
	@echo "=== Cleaning temporary files ==="
	rm -f $(TARGET) $(STUDENT_OBJ)
#--------------------------------------
# Prebuilt tester objects(no source, do Not try implicit .cu->.o rules
#---------------------------------------

tester/tester_iluvatar.o tester/tester_nv.o tester/tester_moore.o tester/tester_metax.o:
	@:


# -------------------------------
# Dependency Rules (Core Logic)
# -------------------------------
# Generate executable: Link kernel code (kernels.o) with test logic (tester.o)
$(TARGET): $(STUDENT_OBJ) $(TEST_OBJ)
	@echo "=== Linking executable (student code + test logic) ==="
#       $(CC) $(CFLAGS) $(PLATFORM_DEFINE) -o $@ $^ $(EXTRA_LIBS)
ifeq ($(PLATFORM),iluvatar)
	$(CC) -std=c++17 -O3 $(IVCORE_LDFLAGS) -o $@ $^ $(EXTRA_LIBS)
else
	$(CC) $(CFLAGS) $(PLATFORM_DEFINE) -o $@ $^ $(EXTRA_LIBS)
endif

# Generate src object: Compile kernels.cu (triggers template instantiation)
$(STUDENT_OBJ): $(STUDENT_SRC)
	@echo "=== Compiling student code ($(STUDENT_SRC)) ==="
#       $(CC) $(CFLAGS) $(PLATFORM_DEFINE) -c $< -o $@
ifeq ($(PLATFORM),iluvatar)
	$(CC) $(IVCORE_CFLAGS) -c $< -o $@
else
	$(CC) $(CFLAGS) $(PLATFORM_DEFINE) -c $< -o $@
endif