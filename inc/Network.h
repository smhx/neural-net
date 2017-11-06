#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <string>
#include <fstream>

class Network {

  private:
	typedef std::vector<double> vdbl;
	typedef std::vector<vdbl> v2dbl;
	typedef std::vector<v2dbl> v3dbl;
	typedef std::pair<vdbl, vdbl> trdata; // training data
	typedef std::vector<trdata> trbatch;
  public:

	Network(const std::vector<int>& sizes);

	Network(std::string fname);

	void SGD(trbatch& data, int numEpochs, int batchSize, double maxRate, double minRate, trbatch& test);
	void feedForward(vdbl& inputLayer); // pass by reference. input layer will output as output layer

	void write(std::string fname);

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

	// the number of layers in the network
	int numLayers;

	// how quickly it learns
	double learningRate, minLearningRate, maxLearningRate;

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

	// Random stuff 

	// random device class instance, source of 'true' randomness for initializing random seed
	std::random_device randDev; 

    // Mersenne twister PRNG, initialized with seed from previous random device instance
	std::mt19937 randGen;

	// normal distribution
    std::normal_distribution<double> randDistribution; 
};

#endif