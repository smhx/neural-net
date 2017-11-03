#include "../inc/Network.h"

Network::Network(const vector<int>& sizes) {

	randGen = mt19937(randDev()); 

	// defaults to mean of 0.0, standard dev of 1.0
    randDistribution = normal_distribution<double>(); 

	layerSizes = sizes;

	biases = v2dbl(sizes.size());
	weights = v3dbl(sizes.size()-1);

	for (int i = 0; i < sizes.size(); ++i) {
		// don't set for input layer
		if (i) { 
			biases[i] = vdbl(sizes[i]);
			for (int j = 0; j < biases[i].size(); ++j) {
				biases[i][j] = randDistribution(randGen); // is a random double
			}
		}
		
		weights[i] = v2dbl(sizes[i]);
		for (int j = 0; j < weights[i].size(); ++j) {
			weights[i][j] = vdbl(sizes[i+1]);
			for (int k = 0; k < weights[i][j].size(); ++k) {
				weights[i][j][k] = randDistribution(randGen);
			}
		}
	}
}

Network::vdbl Network::feedForward(vdbl& a) {
	vdbl activation=a, dot;
	for (int i = 0; i + 1 < layerSizes.size(); ++i) {
		// update layer i + 1 from layer i
		dot = vdbl(0.0, layerSizes[i+1]);
		for (int j = 0; j < layerSizes[i]; ++j) {
			for (int k = 0; k < layerSizes[i+1]; ++k) {
				dot[k] += weights[i][j][k] * activation[j];
			}
		}
		activation.resize(dot.size());
		for (int k = 0; k < activation.size(); ++k) {
			activation[k] = sigmoid(dot[k]+biases[i+1][k]);
		}
	}
	return activation;
}


void Network::SGD(vector<Data>& data, int numEpochs, int batchSize, double trainingRate) {
	for (int epoch = 1; epoch <= numEpochs; ++epoch) {
		shuffle(data.begin(), data.end(), randGen);
		vector< vector<Data> > batches;
		for (int i = 0; i < data.size(); ++i) {
			if (i%batchSize==0) batches.push_back(vector<Data>());
			batches.back().push_back(data[i]);
		}
		for (auto batch : batches) {
			updateBatch(batch, trainingRate);
			printf("Epoch %d complete\n", epoch);
		}
	}
}

void Network::updateBatch(vector<Data>& batch, double trainingRate) {
	v3dbl gradb(biases.size() ), gradw(weights.size());
	for (int i = 0; i < biases.size(); ++i) {
		gradb[i]=v2dbl(biases[i].size());
		//...
	}
}

inline double Network::sigmoid(double x) {return 1.0 / (1.0 + exp(-x));}
inline double Network::sigmoidPrime(double x) {return sigmoid(x)*(1-sigmoid(x));}
