#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <cmath>

#include "Data.h"

using namespace std;

class Network {

private:
	typedef vector<double> vdbl;
	typedef vector<vdbl> v2dbl;
	typedef vector<v2dbl> v3dbl;

public:

	Network(const vector<int>& sizes);
	void SGD(vector<Data>& data, int numEpochs, int batchSize, double trainingRate);
	vdbl feedForward(vdbl& inputLayer);

private: // methods

	double sigmoid(double x);

	double sigmoidPrime(double x);

	void updateBatch(vector<Data>& batch, double trainingRate);

private: // properties

	// the size of layer i with layer 0 = input layer
	vector<int> layerSizes;

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
	random_device randDev; 

    // Mersenne twister PRNG, initialized with seed from previous random device instance
	mt19937 randGen;

	// normal distribution
    normal_distribution<double> randDistribution; 


};

#endif