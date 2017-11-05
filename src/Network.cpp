#include "../inc/Network.h"

using namespace std; // does not affect main.cpp

typedef std::vector<double> vdbl;
typedef std::pair<vdbl, vdbl> trdata;

// initialize the network with random weights and biases
Network::Network(const vector<int>& sizes) {

	randGen = mt19937(randDev()); 

	// defaults to mean of 0.0, standard dev of 1.0
	randDistribution = normal_distribution<double>(); 

	numLayers = sizes.size();

	layerSizes = sizes;

	biases = v2dbl(sizes.size());
	weights = v3dbl(sizes.size()-1);

	for (int i = 0; i < numLayers; ++i) {
		// don't set for input layer
		if (i) { 
			biases[i] = vdbl(sizes[i]);
			for (int j = 0; j < biases[i].size(); ++j) {
				biases[i][j] = randDistribution(randGen); // is a random double
			}
		}
		// don't set for output layer
		if (i < numLayers - 1)
		{
			weights[i] = v2dbl(sizes[i]);
			for (int j = 0; j < weights[i].size(); ++j)	{
				weights[i][j] = vdbl(sizes[i + 1]);
				for (int k = 0; k < weights[i][j].size(); ++k) {
					weights[i][j][k] = randDistribution(randGen);// / sqrt(sizes[i]);
				}
			}
		}
	}
}

void Network::feedForward(vdbl& a) {
	// a is initially the vector of inputs
	vdbl dot;
	for (int i = 0; i < numLayers - 1; ++i) {
		// update layer i + 1 from layer i
		dot = vdbl(layerSizes[i+1], 0.0);
		for (int j = 0; j < layerSizes[i]; ++j) {
			for (int k = 0; k < layerSizes[i+1]; ++k) {
				dot[k] += weights[i][j][k] * a[j];
			}
		}
		a.resize(dot.size());
		for (int k = 0; k < a.size(); ++k) {
			a[k] = sigmoid(dot[k]+biases[i+1][k]);
		}
	}
}

void Network::SGD(trbatch& data, int numEpochs, int batchSize, double trainingRate, trbatch& test) {
	for (int epoch = 1; epoch <= numEpochs; ++epoch) {
		shuffle(data.begin(), data.end(), randGen);
		vector< trbatch > batches;
		for (int i = 0; i < data.size(); ++i) {
			if (i%batchSize == 0) batches.push_back(trbatch());
			batches.back().push_back(data[i]);
		}
		for (auto batch : batches) {
			updateBatch(batch, trainingRate);
		}
		printf("Epoch %d: ", epoch);
		testBatch(test);
	}
}

void Network::updateBatch(const trbatch& batch, double trainingRate) {

	// gradb[i][j] is the gradient for the bias of the jth node in layer i
	// gradw[i][j][k] is the gradient for the weight from the jth node in layer i to the kth node in layer i+1
	v2dbl gradb(numLayers), dgradb(numLayers);
	for (int i = 0; i < numLayers; ++i) {
		gradb[i] = dgradb[i] = vdbl(layerSizes[i], 0.0);
	}
	v3dbl gradw(numLayers - 1), dgradw(numLayers - 1);
	for (int i = 0; i < numLayers - 1; ++i)	{
		gradw[i] = dgradw[i] = v2dbl(layerSizes[i]);
		for (int j = 0; j < layerSizes[i]; ++j)	{
			gradw[i][j] = dgradw[i][j] = vdbl(layerSizes[i + 1], 0.0);
		}
	}
	// Range based loops by constant reference!
	for (const trdata& data : batch) { 

		// dgradb[i][j] is the partial derivative of the cost function 
		// for the current data relative to b[i][j]
		// dgradw[i][j][k] is the partial derivative of the cost function
		// for the current data relative to w[i][j][k]

		// pass dgradb and dgradw by reference
		backprop(data, dgradb, dgradw); 


		for (int i = 0; i < gradb.size(); ++i) {
			for (int j = 0; j < gradb[i].size(); ++j) {
				gradb[i][j] += dgradb[i][j];
			}
		}
		for (int i = 0; i < gradw.size(); ++i) {
			for (int j = 0; j < gradw[i].size(); ++j) {
				for (int k = 0; k < gradw[i][j].size(); ++k) {
					gradw[i][j][k] += dgradw[i][j][k];
				}
			}
		}
	}

	for (int i = 0; i < biases.size(); ++i) {
		for (int j = 0; j < biases[i].size(); ++j) {
			biases[i][j] -= trainingRate/static_cast<double>(batch.size()) * gradb[i][j];
		}
	}

	for (int i = 0; i < weights.size(); ++i) {
		for (int j = 0; j < weights[i].size(); ++j) {
			for (int k = 0; k < weights[i][j].size(); ++k) {
				weights[i][j][k] -= trainingRate/static_cast<double>(batch.size()) * gradw[i][j][k];
			}
		}
	}

}

void Network::backprop(const trdata& data, v2dbl& dgradb, v3dbl& dgradw)
{
	vdbl a = data.first;
	v2dbl activations(numLayers);
	activations[0] = a;
	v2dbl zs(numLayers);

	// feedforward
	for (int i = 0; i < numLayers - 1; ++i)
	{
		// calculate a and z of layer i+1 from layer i
		// a = sigma(z)
		vdbl z = biases[i + 1];
		for (int j = 0; j < layerSizes[i]; ++j) {
			for (int k = 0; k < layerSizes[i + 1]; ++k)	{
				z[k] += weights[i][j][k] * a[j];
			}
		}
		zs[i + 1] = z;
		a.resize(layerSizes[i + 1]);
		a = sigmoid(z);
		activations[i + 1] = a;
	}

	// backpropagate
	vdbl delta(layerSizes[numLayers - 1]);
	for (int i = numLayers - 1; i > 0; --i) {
		if (i == numLayers - 1)	{
			// calculate delta of last layer
			vdbl sp = sigmoidPrime(zs[i]);
			vdbl x = costDerivative(activations[i], data.second);
			// delta = multiply(x, sp); //for quadratic cost
			delta = x; // for cross-entropy cost
		}
		else {
			// calculate delta of layer i from layer i+1
			vdbl sp = sigmoidPrime(zs[i]);
			vdbl x(layerSizes[i], 0.0);
			for (int j = 0; j < layerSizes[i]; ++j)	{
				for (int k = 0; k < layerSizes[i + 1]; ++k)	{
					x[j] += weights[i][j][k] * delta[k];
				}
			}
			delta = multiply(x, sp);
		}

		// calculate change for bias[i] and weights[i-1]
		dgradb[i] = delta;
		for (int j = 0; j < layerSizes[i-1]; ++j) {
			for (int k = 0; k < layerSizes[i]; ++k)	{
				dgradw[i-1][j][k] = activations[i-1][j] * delta[k];
			}
		}
	}
	return;
}

void Network::testBatch(const trbatch& batch) {
	int count = 0;
	double cost = 0.0;
	for (trdata data : batch) {
		vdbl in = data.first, out = data.second;
		feedForward(in);
		//check if correct
		double max = in[0]; int index = 0;
		for (int i = 0; i < in.size(); ++i) {
			if (in[i] > max) {
				max = in[i];
				index = i;
			}
		}
		if (out[index] > 0.9)
			++count;
		//calculate cost
		for (int i = 0; i < in.size(); ++i) {
			// cost += 0.5*(in[i] - out[i])*(in[i] - out[i]) / (batch.size()); for quadratic cost
			cost -= (out[i] * log(in[i]) + (1 - out[i])*log(1 - in[i])) / batch.size();
		}
	}
	printf("%d/%lu correct, cost = %f\n", count, batch.size(), cost);
}

inline double Network::sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

inline double Network::sigmoidPrime(double x) { return sigmoid(x)*(1 - sigmoid(x)); }

inline vdbl Network::sigmoid(const vdbl& x) {
	vdbl y(x.size());
	for (int i = 0; i < x.size(); ++i)
		y[i] = sigmoid(x[i]);
	return y;
}

inline vdbl Network::sigmoidPrime(const vdbl& x) {
	vdbl y(x.size());
	for (int i = 0; i < x.size(); ++i)
		y[i] = sigmoidPrime(x[i]);
	return y;
}

inline vdbl Network::multiply(const vdbl& x, const vdbl& y) {
	vdbl z(x.size());
	for (int i = 0; i < x.size(); ++i)
		z[i] = x[i]*y[i];
	return z;
}

inline vdbl Network::costDerivative(const vdbl& activation, const vdbl& ans) {
	if (activation.size() != ans.size())
		printf("Error in costDerivative: activation.size = %d, ans.size = %d\n", activation.size(), ans.size());
	vdbl x(activation.size());
	for (int i = 0; i < activation.size(); ++i)
		x[i] = activation[i] - ans[i];
	return x;
}