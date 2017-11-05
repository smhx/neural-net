CXX = g++
CXXFLAGS = -std=c++11 -Wall

bin/main: bin/main.o bin/Network.o 
	$(CXX) $(CXXFLAGS) -o bin/main bin/main.o bin/Network.o


bin/main.o: main.cpp inc/Network.h 
	$(CXX) $(CXXFLAGS) -c -o bin/main.o main.cpp 

bin/tester: bin/tester.o bin/Network.o
	$(CXX) $(CXXFLAGS) -o bin/tester bin/tester.o bin/Network.o

bin/tester.o: tests/tester.cpp inc/Network.h
	$(CXX) $(CXXFLAGS) -c -o bin/tester.o tests/tester.cpp 

bin/Network.o: src/Network.cpp inc/Network.h 
	$(CXX) $(CXXFLAGS) -c -o bin/Network.o src/Network.cpp


clean :
	rm bin/*
