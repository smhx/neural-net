#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
#include <functional>

#include "types.h"

class Layer
{

public:

	Layer(int _in, int _out, int _miniBatchSize);

	// converts the output of the previous layer to the output of this layer
	// input is an in x miniBatchSize matrix, each column is a data set
	void apply(Mat& input);

private: // methods

	double sigmoid(double x);

private: // properties

	// sizes of the layer's input and output
	int in, out;

	int miniBatchSize;

	// an out x in matrix of weights
	// weights[i][j] is the weight of node j in the previous layer to node i in the current layer
	Mat weights;

	// an out x 1 vector of biases
	Vec biases;

	// Random stuff 

	// random device class instance, source of 'true' randomness for initializing random seed
	std::random_device randDev;

	// Mersenne twister PRNG, initialized with seed from previous random device instance
	std::mt19937 randGen;

	// normal distribution
	std::normal_distribution<double> randDistribution;
};

#endif
