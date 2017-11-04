#include <cstdio>
#include <vector>
#include <ctime>
#include <cstdlib>

#include "inc/Network.h"

using namespace std;

typedef std::vector<double> vdbl;
typedef std::pair<vdbl, vdbl> trdata;

vdbl binary(int i, int bits)
{
	vdbl v(bits, 0.0);
	for (int j = 0; j < bits; ++j) {
		if (i&(1 << j))
			v[j] = 1.0;
	}
	return v;
}

int main() {
	srand(time(NULL));
	int bits = 10;
	vector<int> sizes({ bits, 10*bits, 2*bits });
	Network n(sizes);
	vector<trdata> training(1000), testing(10);
	vector<int> inTesting;
	for (trdata& data : testing) {
		int num = rand() & ((1 << bits) - 1);
		data.first = binary(num, bits);
		data.second = binary(num*num, 2 * bits);
		inTesting.push_back(num);
	}
	for (trdata& data : training) {
		int num;
		do {
			num = rand() & ((1 << bits) - 1);
		} while (find(inTesting.begin(), inTesting.end(), num) != inTesting.end());
		data.first = binary(num, bits);
		data.second = binary(num*num, 2*bits);
	}
	n.SGD(training, 100, 10, 3, testing);
}