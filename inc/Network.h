#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <cmath>

// #define MAXN 100005 all #defines are visible in main.cpp

#include "Data.h"

// using namespace std; Should not use namespace in header file!

class Network {

private:
	typedef std::vector<double> vdbl;
	typedef std::vector<vdbl> v2dbl;
	typedef std::vector<v2dbl> v3dbl;

public:

	Network(const std::vector<int>& sizes);
	void SGD(std::vector<Data>& data, int numEpochs, int batchSize, double trainingRate);
	vdbl feedForward(vdbl& inputLayer);

private: // methods

	double sigmoid(double x);

	double sigmoidPrime(double x);

	void updateBatch(std::vector<Data>& batch, double trainingRate);

	void backprop(Data& data, v2dbl& dgradb, v3dbl& dgradw);

private: // properties

	// the size of layer i with layer 0 = input layer
	std::vector<int> layerSizes;

	// size of weights is number of layers - 1
	// each element of weights ia a weight matrix from layer i to layer i+1
	// so weights[i] is a layerSizes[i] x layerSizes[i+1] matrix
	v3dbl weights; 

	// bias of layer i
	// skip i = 0 cuz input has no bias
	// has size of layerSizes.size()
	v2dbl biases;

	// Random stuff

	// random device class instance, source of 'true' randomness for initializing random seed
	std::random_device randDev; 

    // Mersenne twister PRNG, initialized with seed from previous random device instance
	std::mt19937 randGen;

	// normal distribution
    std::normal_distribution<double> randDistribution; 
};

#endif