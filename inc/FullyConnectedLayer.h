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

	static void dothing();

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

#define FC_LAYER_TEMPLATE template<typename ActivationFn>

using namespace std;

FC_LAYER_TEMPLATE 
void FullyConnectedLayer<ActivationFn>::dothing() {
	printf("hey\n");
}

FC_LAYER_TEMPLATE
FullyConnectedLayer<ActivationFn>::FullyConnectedLayer() {}

// currently, this is just a fully connected layer using sigmoid activation function
FC_LAYER_TEMPLATE
FullyConnectedLayer<ActivationFn>::FullyConnectedLayer(int _in, int _out) {
	in = _in;
	out = _out;

	// random
	randGen = mt19937(chrono::high_resolution_clock::now().time_since_epoch().count());
	// defaults to mean of 0.0, standard dev of 1.0
	randDistribution = normal_distribution<double>();
	
	weights.resize(out, in);
	biases.resize(out);

	for (int i = 0; i < out; ++i) {
		// set random weights
		for (int j = 0; j < in; ++j)
			weights(i, j) = randDistribution(randGen) / sqrt(in);
		// set random biases
		biases(i) = randDistribution(randGen);
	}
}

// destructor
FC_LAYER_TEMPLATE
FullyConnectedLayer<ActivationFn>::~FullyConnectedLayer(){
	printf("FullyConnectedLayer destructor called\n");
}

FC_LAYER_TEMPLATE
void FullyConnectedLayer<ActivationFn>::apply(Mat& input) {
	prevActivations = input; // this is a^(l-1) in the tutorial
	pre = weights*input + biases.replicate(1, input.cols()); // these are the z-values in the tutorial
	activations = pre.unaryExpr(&ActivationFn::activation); // a = sigma(z) in tutorial		//this gives an error when activation isn't static
	derivs = pre.unaryExpr(&ActivationFn::activationDeriv); // this is sigma'(z) in tutorial	//same thing
	input = activations; // changes input directly, since it is passed by reference
}

// WTD is W^T x D, where W^T is the transpose of weight matrix, D is delta vector
FC_LAYER_TEMPLATE
void FullyConnectedLayer<ActivationFn>::computeDeltaLast(const Mat& output, const Mat& ans, Mat& WTD) {
	delta = costDeriv(output, ans).cwiseProduct(derivs); // delta^L = grad_a(C) * sigma'(z^L)	(BP1)
//	cout << "\nLayer 1: delta is " << delta.rows() << " x " << delta.cols();
	WTD = weights.transpose() * delta; // this is needed to compute delta^(L-1)
}

FC_LAYER_TEMPLATE
void FullyConnectedLayer<ActivationFn>::computeDeltaBack(Mat& WTD) {
	delta = WTD.cwiseProduct(derivs); // delta^l = ((W^(l+1))^T x delta^l) * sigma'(z)		(BP2)
//	cout << "\nLayer 0: delta is " << delta.rows() << " x " << delta.cols();
	WTD = weights.transpose() * delta; // this is needed to compute delta^(l-1)
}

FC_LAYER_TEMPLATE
void FullyConnectedLayer<ActivationFn>::updateBiasAndWeights(double lrate) {
	biases -= lrate*delta.rowwise().mean(); // (BP3)
	weights -= (lrate / delta.cols())*(delta * prevActivations.transpose()); // (BP4)
}

FC_LAYER_TEMPLATE
inline Mat FullyConnectedLayer<ActivationFn>::costDeriv(const Mat& output, const Mat& ans) {
	 return output - ans;
}

FC_LAYER_TEMPLATE
void FullyConnectedLayer<ActivationFn>::print()
{
//	cout << "weights:\n" << weights;
//	cout << "\ndelta:\n" << delta;
//	cout << "\nprevactivation:\n" << prevActivations;
	cout << "\npre\n" << pre;
//	cout << "\nderivs\n" << derivs;
//	cout << "\nBiases:\n" << biases;
}

FC_LAYER_TEMPLATE
inline pair<int, int> FullyConnectedLayer<ActivationFn>::getSize() { return make_pair(in, out); }

#endif
