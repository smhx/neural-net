#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <cstdlib> // random

#include "Data.h"

using namespace std;

class Network {
public:
	Network(const vector<int>& sizes);
	void SGD(const vector<Data>& data, int numEpochs, double trainingRate);
private:

	void updateBatch(vector<Data> batch);

	double trainingRate;

	// the size of layer i with layer 0 = input layer
	vector<int> layerSizes;

	// size of weights is number of layers - 1
	// each element of weights ia a weight matrix from layer i to layer i+1
	// so weights[i] is a layerSizes[i] x layerSizes[i+1] matrix
	vector< vector< vector<double> > > weights; 

	// bias of layer i
	// skip i = 0 cuz input has no bias
	vector< vector<double> > bias;


};

#endif