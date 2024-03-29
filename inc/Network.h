#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
#include <functional>

//#include "types.h"
/*
typedef long long ll;

typedef std::vector<double> vdbl;
typedef std::vector<vdbl> v2dbl;
typedef std::vector<v2dbl> v3dbl;

typedef std::pair<vdbl, vdbl> trdata; // training data
typedef std::vector<trdata> trbatch;

typedef std::function<bool(const vdbl&, const vdbl&) > checker_type;

class Network
{

public:

	Network(const std::vector<int>& sizes, const checker_type& f, int batchSize, double _learnRate, double maxRate, double minRate, double L2, double momentum);

	Network(std::ifstream& fin, const checker_type& f);

	Network& operator=(const Network& net);

	void SGD(trbatch& data, trbatch& test, int numEpochs, std::string fname);
	void feedForward(vdbl& inputLayer); // pass by reference. input layer will output as output layer

										// void write(std::string fname);
	friend std::ofstream& operator<<(std::ofstream& f, const Network& n);

private: // methods

	double sigmoid(double x);
	double sigmoidPrime(double x);
	vdbl sigmoid(const vdbl& x);
	vdbl sigmoidPrime(const vdbl& x);
	vdbl multiply(const vdbl& x, const vdbl& y);
	vdbl costDerivative(const vdbl& activation, const vdbl& ans);

	void updateBatch(const trbatch& batch);

	void backprop(const trdata& trdata, v2dbl& dgradb, v3dbl& dgradw);

	void testBatch(const trbatch& batch);

private: // properties
	checker_type checker;

	// the number of layers in the network
	int numLayers;

	int batchSize;

	// how quickly it learns
	double learnRate, maxLearn, minLearn;

	// how much L2regularization affects cost
	// if high, it will focus on keeping weights low
	// if low, it will focus on minimizing regular cost function
	double L2weight;

	double momentum;

	// to track progress
	double maxfrac = 0;

	// the size of layer i with layer 0 = input layer
	std::vector<int> layerSizes;

	// size of weights is numLayers - 1
	// each element of weights is a weight matrix from layer i to layer i+1
	// so weights[i] is a layerSizes[i] x layerSizes[i+1] matrix
	// weights[i][j][k] is the weight of the edge from the jth node in layer i to the kth node in layer i+1
	v3dbl weights;

	// bias of layer i
	// skip i = 0 cuz input has no bias
	// has size of numLayers
	// bias[i][k] is the bias of the jth node in layer i
	v2dbl biases;

	// the velocity (rate of change) of each weight
	// same size as weights
	v3dbl velocity;

	// Random stuff 

	// random device class instance, source of 'true' randomness for initializing random seed
	std::random_device randDev;

	// Mersenne twister PRNG, initialized with seed from previous random device instance
	std::mt19937 randGen;

	// normal distribution
	std::normal_distribution<double> randDistribution;
};
*/
#endif