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

# Optional sanitizers: set SANITIZE=address or SANITIZE=undefined
SANITIZE ?=
ifneq ($(SANITIZE),)
CFLAGS += -fsanitize=$(SANITIZE) -fno-omit-frame-pointer
LDFLAGS += -fsanitize=$(SANITIZE)
endif

VALGRIND ?= valgrind
VALGRIND_FLAGS ?= --leak-check=full --track-origins=yes
VALGRIND_TEST ?= tests/test_div_exp

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
# register CUDA stub backend for fallback selection when CUDA is disabled
CFLAGS += -DTINYFIN_ENABLE_CUDA_STUB
endif

ifdef ENABLE_BLAS
CFLAGS += -DTINYFIN_ENABLE_BLAS
LDFLAGS += -lopenblas
else
# register BLAS stub backend for fallback selection when BLAS is disabled
CFLAGS += -DTINYFIN_ENABLE_BLAS_STUB
endif

# Always register stub backends for OpenGL/Vulkan so backend selection can be tested.
CFLAGS += -DTINYFIN_ENABLE_OPENGL_STUB -DTINYFIN_ENABLE_VULKAN_STUB

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

asan:
	$(MAKE) clean
	$(MAKE) SANITIZE=address DEBUG=1 all libtinyfin.so

ubsan:
	$(MAKE) clean
	$(MAKE) SANITIZE=undefined DEBUG=1 all libtinyfin.so

valgrind: all
	$(VALGRIND) $(VALGRIND_FLAGS) $(VALGRIND_TEST)

.PHONY: all clean asan ubsan valgrind
