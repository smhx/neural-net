#include <cstdio>
#include <vector>
#include <ctime>
#include <cstdlib>

#include "inc/Network2.h"
#include "inc/Layer.h"
#include "inc/types.h"

#include <Eigen/Dense>

using namespace std;

Vec binary(long long i, int bits)
{
	Vec v(bits, 0.0);
	for (int j = 0; j < bits; ++j) {
		if (i&(1LL << j))
			v[j] = 1.0;
	}
	return v;
}

vdbl mod10(long long i)
{
	vdbl v(10, 0.0);
	v[i % 10] = 1.0;
	return v;
}

bool check(const vdbl& tocheck, const vdbl& correct) {
	if (tocheck.size() != correct.size()) {
		printf("ERROR different size\n");
		return false;
	}
	bool works = true;
	for (int i = 0; i < tocheck.size() && works; ++i) {
		if (abs(tocheck[i]-correct[i]) >= 0.5) works = false;
	}
	return works;
}

int main() {
	srand(time(NULL));
//	ifstream fin("tests/test2.txt");

	int bits = 15;
	int mbs = 8; //mini batch size
	Layer l1(6, 12, mbs);
	Layer l2(12, 12, mbs);
	vector<Layer> layers;
	layers.push_back(l1);
	layers.push_back(l2);
	Network2 n(layers, mbs, 0.1);

	vector<trdata> training(20000), testing(1000);
	
	for (trdata& data : testing) {
		long long i = rand() & ((1 << bits) - 1);
		data.first = binary(i, bits);
		data.second = binary(i*i, bits*2);
	}
	for (trdata& data : training) {
		long long i = rand() & ((1 << bits) - 1);
		data.first = binary(i, bits);
		data.second = binary(i*i, bits*2);
	}

	n.train(training, testing, 10);
}