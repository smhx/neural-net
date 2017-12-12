#include "../inc/Network2.h"
//#include <Eigen/Core>
using namespace std;

// Network2::Network2(const std::vector<Layer>& _layers, const checker_type& ch, int mbs, double lr, double maxr, double minr, double L2, double m)

Network2::Network2(const std::vector<Layer>& _layers, const checker_type& ch, int _in, int _out, int mbs, double lr, double _L2) {
	layers = _layers;
	checker = ch;
	miniBatchSize = mbs;
	learnRate = lr;
//	maxRate = maxr;
//	minRate = minr;
	L2 = _L2;
//	momentum = m;
	numLayers = layers.size();
	randGen = mt19937(randDev());
	in = _in;
	out = _out;
}

void Network2::feedForward(Mat& input) {
	for (int i = 0; i < numLayers; ++i)
		layers[i].apply(input);
}

void Network2::train(trbatch& data, trbatch& test, int numEpochs) {
	for (int epoch = 1; epoch <= numEpochs; ++epoch)
	{
		shuffle(data.begin(), data.end(), randGen);
		Mat batch(in, miniBatchSize);
		Mat answers(out, miniBatchSize);
		for (int i = 0; i < data.size(); ++i) {
			batch.col(i % miniBatchSize) = data[i].first;
			answers.col(i % miniBatchSize) = data[i].second;
			if ((i + 1) % miniBatchSize == 0) {
				// feedforward
				feedForward(batch);

				// backpropagate error
				Mat WTD;
				layers[numLayers - 1].computeDeltaLast(batch, answers, WTD);
				for (int i = numLayers - 2; i >= 0; i--) {
					layers[i].computeDeltaBack(WTD);
				}
				// updates
				for (int i = 0; i < numLayers; i++)	{
					layers[i].updateBiasAndWeights(learnRate, L2);
				}
				batch.resize(in, miniBatchSize);
				answers.resize(out, miniBatchSize);
			}
		}
		/*
		cout << "\nLayer 0 ";
		layers[0].print();
		cout << "\nLayer 1 ";
		layers[1].print();
		*/
		// evaluate progress
		Mat testBatch(in, test.size());
		Mat testAns(out, test.size());
		for (int i = 0; i < test.size(); ++i) {
			testBatch.col(i) = test[i].first;
			testAns.col(i) = test[i].second;
		}
		feedForward(testBatch);
//		cout << testBatch << "\n\n" << testAns;
		if (isnan(testBatch(0, 0)))
		{
			cout << "NAN!\n";
			layers[1].print();
		}

		auto p = checker(testBatch, testAns);
		printf("Epoch %d: %d out of %lu correct, average cost: %.3f\n", epoch, p.first, testBatch.cols(), p.second);
//		layers[0].print();
//		layers[1].print();
//		layers[2].print();
	}
}
