#ifndef NETWORK2_H
#define NETWORK2_H

#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
#include <functional>

#include "types.h"

class Network
{

public:

	Network(const std::vector<int>& sizes, const checker_type& f, int batchSize, double _learnRate, double maxRate, double minRate, double L2, double momentum);

	void SGD(trbatch& data, trbatch& test, int numEpochs, std::string fname);
	void feedForward(vdbl& inputLayer); // pass by reference. input layer will output as output layer

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
};

#endif