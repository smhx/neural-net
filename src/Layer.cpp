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
	prevActivations = input;
	int miniBatchSize = input.cols();
	pre = weights*input + biases.replicate(1, miniBatchSize);
	activations = pre.unaryExpr(&Layer::activation); //this gives an error when activation isn't static
	derivs = pre.unaryExpr(&Layer::activationDeriv); //same thing
	input = activations;
}

// WTD is W^T x D, where W^T is the transpose of weight matrix, D is delta vector
void Layer::computeDeltaLast(Mat& output, Mat& ans, Mat& WTD) {
	
//	cout << "\nBiases:\n" << biases;
//	cout << "\nWeights:\n" << weights;
//	cout << "\nOutput:\n" << output;
//	cout << "\nAnswer:\n" << ans;
	delta = costDeriv(output, ans).cwiseProduct(derivs);
	WTD = weights.transpose() * delta;
}

void Layer::computeDeltaBack(Mat& WTD) {
	delta = WTD.cwiseProduct(derivs);
	WTD = weights.transpose() * delta;
}

void Layer::updateBiasAndWeights(double lrate) {
	
//	cout << "\nBiases:\n" << biases;
//	cout << "\nWeights:\n" << weights;
//	cout << "\nDelta:\n" << delta;
//	cout << "\nDerivs:\n" << derivs;
	
	biases -= lrate*delta.rowwise().mean();
	weights -= lrate*((delta * prevActivations.transpose()).rowwise().mean()).replicate(1, in);
}

double Layer::activation(double x) {
	return 1.0 / (1.0 + exp(-x));
}

inline double Layer::activationDeriv(double x) {
	return activation(x)*(1 - activation(x));
}

inline Mat Layer::costDeriv(Mat& output, Mat& ans) {
	 return ans - output;
}

inline pair<int, int> Layer::getSize() { return make_pair(in, out); }
