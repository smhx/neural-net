#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

struct Data {};

class Network {
public:
	Network(const std::vector<int>& sizes);
	void SGD(const std::vector<Data>& data, int numEpochs, double trainingRate);
private:

};

#endif