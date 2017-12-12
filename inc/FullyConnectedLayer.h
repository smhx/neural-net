#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
#include <functional>
#include <chrono>

#include "types.h"
#include "../inc/ActivationFunction.h"

#include "Layer.h"

template<typename ActivationFn>
class FullyConnectedLayer : public Layer
{

public:
	FullyConnectedLayer();
	FullyConnectedLayer(int _in, int _out);

	~FullyConnectedLayer();

	// converts the output of the previous layer to the output of this layer
	// input is an in x miniBatchSize matrix, each column is a data set
	void apply(Mat& input);

	// if this layer is the last layer, computes the delta (error) given the output and correct answer
	void computeDeltaLast(const Mat& output, const Mat& ans, Mat& WTD);

	// if this layer is not the last layer, computes the delta from the last layer's delta
	void computeDeltaBack(Mat& WTD);

	void updateBiasAndWeights(double lrate);

	std::pair<int, int> getSize();

	void print();
	
// private: // methods

	Mat costDeriv(const Mat& ans, const Mat& output);
	
private: // properties

	// sizes of the layer's input and output
	int in, out;

	// an out x in matrix of weights
	// weights[i][j] is the weight of node j in the previous layer to node i in the current layer
	Mat weights;

	// an out x 1 vector of biases
	Vec biases;

	// the values before the activation function is applied to them
	// these are the z values in the tutorial
	Mat pre;

	// the activations of the layer before it
	// saves a copy of the input from Layer::apply()
	Mat prevActivations;

	// stores the activations to use in backpropagation
	Mat activations;

	// sigma'(z) for each preactivation z
	Mat derivs;

	// the error from the actual answer, used for backpropagation
	// it's an o
	Mat delta;
	
	// Random stuff 
	
	// random device class instance, source of 'true' randomness for initializing random seed
//	std::random_device randDev;
	
	// Mersenne twister PRNG, initialized with seed from previous random device instance
	std::mt19937 randGen;
	
	// normal distribution
	std::normal_distribution<double> randDistribution;
	
};

#endif
