#include <cstdio>
#include <vector>
#include <ctime>
#include <cstdlib>

#include "inc/Network.h"

using namespace std;

typedef std::vector<double> vdbl;
typedef std::pair<vdbl, vdbl> trdata;

vdbl binary(long long i, int bits)
{
	vdbl v(bits, 0.0);
	for (int j = 0; j < bits; ++j) {
		if (i&(1LL << j))
			v[j] = 1.0;
	}
	return v;
}

int main() {
	srand(time(NULL));
	int bits = 12;
	vector<int> sizes({ bits, 4*bits, 2*bits });
	Network n(sizes);
	vector<trdata> training(10 << bits), testing(1000);

	vector<int> inTesting;
	for (trdata& data : testing) {
		long long num = rand() & ((1 << bits) - 1);
		data.first = binary(num, bits);
		data.second = binary(num*num, 2 * bits);
	}
	for (int i = 0; i < 10 << bits; ++i) {
		int num = i & ((1 << bits) - 1);
		training[i].first = binary(num, bits);
		training[i].second = binary(num*num, bits);
	}
	/*
	for (trdata& data : training) {
		long long num = rand() & ((1 << bits) - 1);
		data.first = binary(num, bits);
		data.second = binary(num*num, 2*bits);
	}*/

	n.SGD(training, 10000, 1, 0.5, testing);
}

/*
To make testing and training data mutually exclusive

vector<int> inTesting;
for (trdata& data : testing) {
int num = rand() & ((1 << bits) - 1);
data.first = binary(num, bits);
data.second = binary(num*num, 2 * bits);
inTesting.push_back(num);
}
sort(inTesting.begin(), inTesting.end());
int num = 0;
for (trdata& data : training) {
while (std::binary_search(inTesting.begin(), inTesting.end(), num))
num = (num + 1)&((1 << bits) - 1);
data.first = binary(num, bits);
data.second = binary(num*num, 2*bits);
num = (num + 1)&((1 << bits) - 1);
}
*/