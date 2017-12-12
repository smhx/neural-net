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

using namespace std;

Vec binary(int i, int bits)
{
	Vec v(bits);
	for (int j = 0; j < bits; ++j) {
		if (i&(1 << j))
			v[j] = 1.0;
		else
			v[j] = 0.0;
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
			if (error >= 0.5)
				works = false;
			cost += error*error;
		}
		if (works)
			++count;
	}
	return make_pair(count, cost/tocheck.cols());
}

int main() {
	SigmoidActivationFunction::activation(1.0);
	Layer* l = new FullyConnectedLayer< SigmoidActivationFunction >(1, 2);
	delete l;
	/*
	srand(time(NULL));

	int bits = 10;	

	typedef FullyConnectedLayer<SigmoidActivationFunction> SigLayer;
	
	vector<Layer*> layers;
	layers.push_back(new SigLayer(2 * bits, 5 * bits));
	layers.push_back(new SigLayer(5 * bits, bits + 1));

	Network2 n(layers, check, 2 * bits, bits + 1, 8, 0.5);

	trbatch training(20000), testing(1000);
	
	for (trdata& data : testing) {
		int i = rand() & ((1 << bits) - 1);
		int j = rand() & ((1 << bits) - 1);
		data.first = binary((i << bits) + j, 2 * bits);
		data.second = binary(i + j, bits + 1);
	}
	
	// for (int i = 0; i < (1 << (2*bits)); i++) {
	// 	testing[i].first = binary(i, 2 * bits);
	// 	testing[i].second = binary((i >> bits) + (i & ((1 << bits) - 1)), bits + 1);
	// }

	for (trdata& data : training) {
		int i = rand() & ((1 << bits) - 1);
		int j = rand() & ((1 << bits) - 1);
		data.first = binary((i << bits) + j, 2 * bits);
		data.second = binary(i + j, bits + 1);
	}
//	cout << testing[5].first << endl;
//	cout << testing[5].second << endl;
	n.train(training, testing, 1000);
	*/
}
/*
vdbl binary(long long i, int bits)
{
	vdbl v(bits, 0.0);
	for (int j = 0; j < bits; ++j)
	{
		if (i&(1LL << j))
			v[j] = 1.0;
	}
	return v;
}

bool check(const vdbl& tocheck, const vdbl& correct)
{
	if (tocheck.size() != correct.size())
	{
		printf("ERROR different size\n");
		return false;
	}
	bool works = true;
	for (int i = 0; i < tocheck.size() && works; ++i)
	{
		if (abs(tocheck[i] - correct[i]) >= 0.5) works = false;
	}
	return works;
}

int main()
{
	srand(time(NULL));
//	ifstream fin("tests/test.txt");
	int bits = 8;
	vector<int> sizes({ 2*bits, 5*bits, bits+1 });
	Network n(sizes, check, 10, 1, 1, 0.2, 0, 0);
//	Network n(fin, check);
	vector<trdata> training(20000), testing(1000);

	for (trdata& data : testing) {
		int i = rand() & ((1 << bits) - 1);
		int j = rand() & ((1 << bits) - 1);
		data.first = binary((i << bits) + j, 2 * bits);
		data.second = binary(i + j, bits + 1);
	}
	for (trdata& data : training) {
		int i = rand() & ((1 << bits) - 1);
		int j = rand() & ((1 << bits) - 1);
		data.first = binary((i << bits) + j, 2 * bits);
		data.second = binary(i + j, bits + 1);
	}
	n.SGD(training, testing, 100, "tests/oldNetwork.txt");
}
*/