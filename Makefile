CC ?= gcc
DEBUGFLAGS ?= -g -O0
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -Werror -I. -Isrc -Iinclude/tinyfin
NVCC ?= nvcc
NVCCFLAGS ?= -O2 -Xcompiler -fPIC -I. -Isrc -Iinclude/tinyfin

# Allow building tests with debug symbols by setting DEBUG=1
ifdef DEBUG
CFLAGS := -std=c99 $(DEBUGFLAGS) -Wall -Wextra -Werror -I. -Isrc -Iinclude/tinyfin
NVCCFLAGS := -G -Xcompiler -fPIC -I. -Isrc -Iinclude/tinyfin
endif

LDFLAGS ?= -lm

# Automatically collect all source files
SRC := $(wildcard src/*.c)
CUDA_SRC := $(wildcard src/*.cu)

ifdef ENABLE_CUDA
CUDA_OBJS := $(CUDA_SRC:.cu=.o)
CFLAGS += -DTINYFIN_ENABLE_CUDA
NVCCFLAGS += -DTINYFIN_ENABLE_CUDA
LDFLAGS += -lcudart
else
CUDA_OBJS :=
endif

ifdef ENABLE_BLAS
CFLAGS += -DTINYFIN_ENABLE_BLAS
LDFLAGS += -lopenblas
endif

# Test files: discover all tests in the tests/ directory
TESTS := $(wildcard tests/*.c)

all: $(TESTS:.c=)

$(TESTS:.c=): %: $(SRC) $(CUDA_OBJS) %.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

libtinyfin.so: $(SRC) $(CUDA_OBJS)
	$(CC) -shared -fPIC $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -f $(TESTS:.c=) libtinyfin.so $(CUDA_OBJS)

.PHONY: all clean
