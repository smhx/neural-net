CXX = g++
CXXFLAGS = -std=c++11 -Wall

main: build/main.o build/Network.o 
	$(CXX) $(CXXFLAGS) -o bin/main build/main.o build/Network.o

build/main.o: main.cpp inc/Network.h 
	$(CXX) $(CXXFLAGS) -c -o build/main.o main.cpp 

digit: build/digit.o build/tester.o build/Network.o
	$(CXX) $(CXXFLAGS) -o bin/digit build/digit.o build/tester.o build/Network.o

build/digit.o: test/digit/digit.cpp inc/Network.h test/tester.h
	$(CXX) $(CXXFLAGS) -c -o build/digit.o test/digit/digit.cpp


gender: build/gender.o build/tester.o build/Network.o
	$(CXX) $(CXXFLAGS) -o bin/gender build/gender.o build/tester.o build/Network.o


build/gender.o: test/gender/gender.cpp inc/Network.h test/tester.h
	$(CXX) $(CXXFLAGS) -c -o build/gender.o test/gender/gender.cpp

sum: build/sum.o build/tester.o build/Network.o
	$(CXX) $(CXXFLAGS) -o bin/sum build/sum.o build/tester.o build/Network.o

build/sum.o: test/sum/sum.cpp inc/Network.h test/tester.h
	$(CXX) $(CXXFLAGS) -c -o build/sum.o test/sum/sum.cpp


build/tester.o: test/tester.cpp test/tester.h inc/Network.h
	$(CXX) $(CXXFLAGS) -c -o build/tester.o test/tester.cpp 

build/Network.o: src/Network.cpp inc/Network.h 
	$(CXX) $(CXXFLAGS) -c -o build/Network.o src/Network.cpp


clean :
	rm bin/*
	rm build/*
