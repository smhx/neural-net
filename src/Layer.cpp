#include "../inc/Layer.h"

using namespace std;

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
	input = weights*input + biases.replicate(1, miniBatchSize);
	input.unaryExpr(&sigmoid);
}

inline double Layer::sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }