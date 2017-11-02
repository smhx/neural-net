#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

#include "Data.h"

class Network {
public:
	Network(const std::vector<int>& sizes);
	void SGD(const std::vector<Data>& data, int numEpochs, double trainingRate);
private:
	
};

#endif