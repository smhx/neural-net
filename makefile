CXX = g++
CXXFLAGS = -std=c++11 -Wall

bin/main: bin/main.o bin/Network.o 
	$(CXX) $(CXXFLAGS) -o bin/main bin/main.o bin/Network.o

bin/main.o: main.cpp inc/Network.h 
	$(CXX) $(CXXFLAGS) -c -o bin/main.o main.cpp 

bin/Network.o: src/Network.cpp inc/Network.h 
	$(CXX) $(CXXFLAGS) -c -o bin/Network.o src/Network.cpp

clean :
	rm bin/*
