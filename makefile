IDIR=inc
CC=g++
CFLAGS=-std=c++11 -I $(IDIR)

ODIR=build
LDIR = Eigen

SRCDIR=src

LIBS=-lm

_DEPS = ActivationFunction.h FullyConnectedLayer.h Layer.h Network2.h types.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = ActivationFunction.o FullyConnectedLayer.o Layer.o Network2.o types.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)


main: $(OBJ) build/main.o
	$(CC) -o main $(OBJ) build/main.o $(CFLAGS)


build/main.o: $(OBJ) main.cpp
	$(CC) -c main.cpp -o build/main.o $(CFLAGS)


.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(IDIR)/*~ 
