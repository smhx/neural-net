#include <cstdio>
#include <vector>
#include <ctime>
#include <cstdlib>

#include "inc/Network.h"
#include "inc/Data.h"

using namespace std;

const int MAX = 100;

int f(int x) {
	return x+1;
}

int main() {
	srand(time(NULL));

	vector<Data> data;
	vector<int> layerSizes;

	int nLayers, numEpochs, batchSize;
	double trainingRate;

	scanf("%d", &nLayers);
	for (int i = 0; i < nLayers; ++i) {
		int sz; scanf("%d", &sz);
		layerSizes.push_back(sz);
	}

	for (int i = 0; i < layerSizes[0]; ++i) {
		int x = rand() % MAX;
		data.push_back(Data(x, f(x) ));
	}

	scanf("%d %d %lf", &numEpochs, &batchSize, &trainingRate);

	Network AI(layerSizes);
	AI.SGD(data, numEpochs, batchSize, trainingRate);
	return 0;
}