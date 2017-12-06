#include "../inc/Layer.h"

using namespace std;

// currently, this is just a fully connected layer using sigmoid activation function
Layer::Layer(int _in, int _out) {
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
			weights(i,j) = randDistribution(randGen) / sqrt(in);
		// set random biases
		biases(i) = randDistribution(randGen);
	}
}

void Layer::apply(Mat& input) {
	prevActivations = input; // this is a^(l-1) in the tutorial
	int miniBatchSize = input.cols();
	pre = weights*input + biases.replicate(1, miniBatchSize); // these are the z-values in the tutorial
	activations = pre.unaryExpr(&Layer::activation); // a = sigma(z) in tutorial		//this gives an error when activation isn't static
	derivs = pre.unaryExpr(&Layer::activationDeriv); // this is sigma'(z) in tutorial	//same thing
	input = activations; // changes input directly, since it is passed by reference
}

// WTD is W^T x D, where W^T is the transpose of weight matrix, D is delta vector
void Layer::computeDeltaLast(const Mat& output, const Mat& ans, Mat& WTD) {
	delta = costDeriv(output, ans).cwiseProduct(derivs); // delta^L = grad_a(C) * sigma'(z^L)	(BP1)
	WTD = weights.transpose() * delta; // this is needed to compute delta^(L-1)
}

void Layer::computeDeltaBack(Mat& WTD) {
	delta = WTD.cwiseProduct(derivs); // delta^l = ((W^(l+1))^T x delta^l) * sigma'(z)		(BP2)
	WTD = weights.transpose() * delta; // this is needed to compute delta^(l-1)
}

void Layer::updateBiasAndWeights(double lrate) {
	biases -= lrate*delta.rowwise().mean(); // (BP3)
	weights -= (lrate / delta.cols())*(delta * prevActivations.transpose()); // (BP4)
}

double Layer::activation(double x) {
	return 1.0 / (1.0 + exp(-x));
}

inline double Layer::activationDeriv(double x) {
	return activation(x)*(1 - activation(x));
}

inline Mat Layer::costDeriv(const Mat& output, const Mat& ans) {
	 return ans - output;
}

inline pair<int, int> Layer::getSize() { return make_pair(in, out); }
