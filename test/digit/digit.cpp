#include "../tester.h"
#include <fstream>
using namespace std;

const int ndigits=10;

int imgsize, ntraining, ntest;

void read(Tester& tester) {
	
	ifstream fin("test/digit/imgdata.txt");

	fin >> imgsize >> ntraining >> ntest;

	tester.training = trbatch(ntraining);

	for (int i = 0; i < ntraining; ++i) {
		for (int j = 0; j < imgsize * imgsize; ++j) {
			double v; fin >> v;
			tester.training[i].first.push_back(v);
		}
		int dig; fin >> dig;
		tester.training[i].second = vdbl(ndigits, 0.0);
		tester.training[i].second[dig] = 1.0;
	}

	tester.testing = trbatch(ntest);

	for (int i = 0; i < ntest; ++i) {
		for (int j = 0; j < imgsize * imgsize; ++j) {
			double v; fin >> v;
			tester.testing[i].first.push_back(v);
		}
		int dig; fin >> dig;
		tester.testing[i].second = vdbl(10, 0.0);
		tester.testing[i].second[dig] = 1.0;
	}	
}

void train(){
	Tester tester;
	read(tester); // reads tester.training and tester.testing
	tester.sizes = {imgsize, 100, ndigits};

	tester.numEpochs = 200;
	tester.batchSize = 20;

	tester.lrate = 100.0;
	tester.maxRate = 150.0;
	tester.minRate = 30.0;
	
	tester.checker = largestCheck;

	tester.write = true;

	tester.train();
}

int main() {
	train();
}