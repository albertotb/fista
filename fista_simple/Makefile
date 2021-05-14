CC=gcc
BIN=fista
PREFIX=/scratch/gaa/local
LDIR=$(PREFIX)/lib
CFLAGS=-std=c11 -O3 -fPIC
LFLAGS=-L$(LDIR) -lm -fopenmp
BLAS=0

ifeq ($(BLAS), 1)
	CFLAGS+=-D_BLAS_
	LFLAGS+=-lopenblas
endif

DEPS=$(wildcard *.h)
SRC=$(wildcard *.c)
OBJ=$(SRC:%.c=%.o)

all: $(BIN)

%.o: %.c $(DEPS)
		$(CC) -c -o $@ $< $(CFLAGS)

$(BIN): $(OBJ)
		$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)
		@rm -rf *.o

.PHONY: clean

clean:
		rm -f $(BIN) *~
