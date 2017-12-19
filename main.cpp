#include <cstdio>
#include <vector>
#include <ctime>
#include <cstdlib>

#include "inc/Network2.h"
#include "inc/Layer.h"
#include "inc/types.h"
#include "inc/FullyConnectedLayer.h"
#include "inc/ActivationFunction.h"

#include <Eigen/Dense>

typedef long long ll;

using namespace std;

Vec binary(ll i, int bits)
{
	Vec v(bits);
	for (int j = 0; j < bits; ++j) {
		if (i & (1LL << j))
			v[j] = 1.0;
		else
			v[j] = -1.0;
	}
	return v;
}

vdbl mod10(long long i)
{
	vdbl v(10, 0.0);
	v[i % 10] = 1.0;
	return v;
}

pair<int,double> check(const Mat& tocheck, const Mat& correct) {
	if (tocheck.rows() != correct.rows() || tocheck.cols() != correct.cols()) {
		printf("ERROR in check: Vectors are different sizes\n");
		return make_pair(0,0);
	}
//	cout << tocheck.col(0) << endl;
//	cout << correct.col(0) << endl;
//	cout << tocheck << endl;
	int count = 0;
	double cost = 0.0;
	for (int col = 0; col < tocheck.cols(); col++)
	{
		bool works = true;
		for (int i = 0; i < tocheck.rows(); ++i) {
			double error = abs(tocheck(i, col) - correct(i, col));
			if (error >= 1.0)
				works = false;
			cost += error*error;
		}
		if (works)
			++count;
	}
	if (isnan(cost))
	{
//		cout << "Cost is nan!\nTest Batch:\n" << tocheck << "\nAnswers\n" << correct;
	}
	return make_pair(count, cost/tocheck.cols());
}

int main() {

	srand(time(NULL));

	int bits = 11;	

	typedef FullyConnectedLayer<SigmoidActivationFunction> SigLayer;
	
	vector<Layer*> layers;
	layers.push_back(new SigLayer(2 * bits, 5 * bits));
	layers.push_back(new SigLayer(5 * bits, bits + 1));

	Network2 n(layers, check, 2 * bits, 2 * bits, 8, 0.001);

	trbatch training(100000), testing(1000);
	
	for (trdata& data : testing) {
		ll i = rand() & ((1 << bits) - 1);
		ll j = rand() & ((1 << bits) - 1);
		data.first = binary((i << bits) + j, 2 * bits);
		data.second = binary(i * j, 2 * bits);
	}
	for (int i = 0; i < (1 << (2*bits)); i++) {
		testing[i].first = binary(i, 2 * bits);
		testing[i].second = binary((i >> bits) + (i & ((1 << bits) - 1)), bits + 1);
	}
		
	for (trdata& data : training) {
		ll i = rand() & ((1 << bits) - 1);
		ll j = rand() & ((1 << bits) - 1);
		data.first = binary((i << bits) + j, 2 * bits);
		data.second = binary(i * j, 2 * bits);
	}
	n.train(training, testing, 500);
}
