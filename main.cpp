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

vdbl mod10(long long i)
{
	vdbl v(10, 0.0);
	v[i % 10] = 1.0;
	return v;
}

int main() {
	srand(time(NULL));
	int bits = 10;
	vector<int> sizes({ bits, bits*32, 2*bits });
	Network n(sizes);
	vector<trdata> training(1<<bits), testing(100);

	for (trdata& data : testing) {
		long long i = rand() & ((1 << bits) - 1);
		data.first = binary(i, bits);
		// data.second = mod10(i);
		data.second = binary(i*i, 2*bits);
	}/*
	for (trdata& data : training) {
		long long i = rand() & ((1 << bits) - 1);
		data.first = binary(i, bits);
		data.second = mod10(i);
	}*/
	for (int i = 0; i < 1 << bits; ++i) {
		training[i].first = binary(i, bits);
		// training[i].second = mod10(i);
		training[i].second = binary(i*i, 2*bits);
	}

	n.SGD(training, 800, 20, 3, testing);
	n.write("tests/test.txt");
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