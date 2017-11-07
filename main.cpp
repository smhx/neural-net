#include <cstdio>
#include <vector>
#include <ctime>
#include <cstdlib>

#include "inc/Network.h"
#include "inc/types.h"

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

	int bits = 10;
	vector<int> sizes({ 2*bits, 15*bits, 2*bits });
//	Network n(sizes, check, 10, 1, 1, 0.2, 0);
	Network n("tests/test.txt", check);

	vector<trdata> training(100000), testing(100);
	
	for (trdata& data : testing) {
		long long i = rand() & ((1 << bits) - 1);
//		long long j = rand() & ((1 << bits) - 1);
		data.first = binary((i << bits) | i, 2 * bits);
		data.second = binary(i*i, bits*2);
	}
	for (trdata& data : training) {
		long long i = rand() & ((1 << bits) - 1);
		data.first = binary((i << bits) | i, 2 * bits);
		data.second = binary(i*i, bits*2);
	}
	/*
	int ind = 0;
	for (int i = 0; i < 1 << bits; ++i) {
		for (int j = 0; j < 1 << bits; ++j)	{
			training[ind].first = binary((i << bits) | j, 2 * bits);
			training[ind++].second = binary(i + j, bits + 1);
		}
	}*/

	n.SGD(training, testing, 10);
	ofstream fout("tests/test.txt");
	fout << n;
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