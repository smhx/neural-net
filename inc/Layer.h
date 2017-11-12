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

	Layer(int inSize, int outSize, int miniBatchSize);

	//converts the output of the previous layer to the output of this layer
	void apply(Mat& in);

private: // methods

	double sigmoid(double x);

private: // properties

	int miniBatchSize;

	// an outSize x inSize matrix of weights
	v3dbl weights;

	// an outSize x 1 vector of biases
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
