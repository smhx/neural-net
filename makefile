CXX = g++
CXXFLAGS = -std=c++11 -Wall

bin/main: bin/main.o bin/Network.o bin/Data.o
	$(CXX) $(CXXFLAGS) -o bin/main bin/main.o bin/Network.o bin/Data.o

bin/main.o: main.cpp inc/Network.h inc/Data.h
	$(CXX) $(CXXFLAGS) -c -o bin/main.o main.cpp 

bin/Network.o: src/Network.cpp inc/Network.h inc/Data.h
	$(CXX) $(CXXFLAGS) -c -o bin/Network.o src/Network.cpp

bin/Data.o: src/Data.cpp inc/Data.h
	$(CXX) $(CXXFLAGS) -c -o bin/Data.o src/Data.cpp

clean :
	rm bin/*
