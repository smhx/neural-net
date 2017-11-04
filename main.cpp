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
	int bits = 20;
	vector<int> sizes({ bits, 2*bits, 2*bits });
	Network n(sizes);
	vector<trdata> training(10000), testing(1000);
	for (trdata& data : training) {
		int num = rand() & ((1 << bits) - 1);
		data.first = binary(num, bits);
		data.second = binary(num*num, 2*bits);
	}
	for (trdata& data : testing)	{
		int num = rand() & ((1 << bits) - 1);
		data.first = binary(num, bits);
		data.second = binary(num*num, 2 * bits);
	}
	n.SGD(training, 100, 10, 1);
	n.testBatch(testing);
}