#include "../inc/Layer.h"

using namespace std;

// currently, this is just a fully connected layer using sigmoid activation function
Layer::Layer(int _in, int _out, int _miniBatchSize) {
	in = _in;
	out = _out;
	miniBatchSize = _miniBatchSize;

	// random
	randGen = mt19937(randDev());
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
	pre = weights*input + biases.replicate(1, miniBatchSize);
	activations = pre.unaryExpr(&Layer::activation);
	derivs = pre.unaryExpr(&Layer::activationDeriv);
	input = activations;
}

// WTD is W^T x D, where W^T is the transpose of weight matrix, D is delta vector
void Layer::computeDeltaLast(Mat& output, Mat& ans, Mat& WTD) {
	delta = costDeriv(output, ans).cwiseProduct(derivs.replicate(1, miniBatchSize));
	WTD = weights.transpose() * delta;
}

void Layer::computeDeltaBack(Mat& WTD) {
	delta = WTD.cwiseProduct(derivs.replicate(1, miniBatchSize));
	WTD = weights.transpose() * delta;
}

void Layer::updateBiasAndWeights(double lrate) {
	biases -= lrate*delta.rowwise().mean();
	weights -= lrate*(delta * prevActivations.transpose()).rowwise().mean();
}



inline double Layer::activation(double x) {
	return 1.0 / (1.0 + exp(-x));
}

inline double Layer::activationDeriv(double x) {
	return activation(x)*(1 - activation(x));
}

inline Mat Layer::costDeriv(Mat& output, Mat& ans) {
	 return ans - output;
}

inline pair<int, int> Layer::getSize() { return make_pair(in, out); }