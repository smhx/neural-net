#include "../inc/Network2.h"
#include <Eigen/Core>
using namespace std;

// Network2::Network2(const std::vector<Layer>& _layers, const checker_type& ch, int mbs, double lr, double maxr, double minr, double L2, double m)

Network2::Network2(const std::vector<Layer>& _layers, int mbs, double lr) {
	layers = _layers;
//	checker = ch;
	miniBatchSize = mbs;
	learnRate = lr;
//	maxRate = maxr;
//	minRate = minr;
//	L2weight = L2;
//	momentum = m;
	numLayers = layers.size();
	randGen = mt19937(randDev());
}

void Network2::feedForward(Mat& input) {
	for (int i = 0; i < numLayers; ++i)
		layers[i].apply(input);
}

void Network2::train(trbatch& data, trbatch& test, int numEpochs) {
	for (int epoch = 1; epoch <= numEpochs; ++epoch)
	{
		shuffle(data.begin(), data.end(), randGen);
		Mat batch(layers[0].getSize().first, miniBatchSize);
		Mat answers(layers[numLayers - 1].getSize().second, miniBatchSize);
		for (int i = 0; i < data.size(); ++i) {
			batch.col(i % miniBatchSize) = data[i].first;
			answers.col(i % miniBatchSize) = data[i].second;
			if ((i + 1) % miniBatchSize == 0) {
				// feedforward
				for (int i = 0; i < numLayers; i++)	{
					layers[i].apply(batch);
				}
				// backpropagate error
				Mat WTD;
				layers[numLayers - 1].computeDeltaLast(batch, answers, WTD);
				for (int i = numLayers - 2; i >= 0; i--) {
					layers[i].computeDeltaBack(WTD);
				}
				// updates
				for (int i = 0; i < numLayers; i++)	{
					layers[i].updateBiasAndWeights(learnRate);
				}
			}
		}
	}
}