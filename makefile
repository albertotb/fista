CC=gcc
BIN=fista
SDIR=./src
IDIR=./include
PREFIX=/scratch/gaa/local
LDIR=$(PREFIX)/lib
# -g is for gdb (debugging) and -pg is for gprof (profiling)
DEBUG=-g
BLAS=1

ifeq ($(BLAS), 1)
	CFLAGS=$(DEBUG) -O3 -I$(IDIR) -D_BLAS_
	LFLAGS=$(DEBUG) -L$(LDIR) -lopenblas -lm -fopenmp
else
	CFLAGS=$(DEBUG) -O3 -I$(IDIR) #-D_PRECOMP_
	LFLAGS=$(DEBUG) -L$(LDIR) -lm -fopenmp
endif

DEPS=$(wildcard $(IDIR)/*.h)
SRC=$(wildcard $(SDIR)/*.c)
OBJ=$(SRC:$(SDIR)/%.c=%.o)

all: $(BIN)

%.o: $(SDIR)/%.c $(DEPS)
		$(CC) -c -o $@ $< $(CFLAGS)

$(BIN): $(OBJ)
		$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)
		@rm -rf *.o

.PHONY: clean

clean:
		rm -f $(BIN) $(SDIR)/*~ $(IDIR)/*~

install:
		ln -s $(BIN) $(PREFIX)
