#include "tester.h"
using namespace std;

vdbl tobinary(ll x, int bits) {
	vdbl v(bits, 0.0);
	for (int i = 0; i < bits; ++i) if (x & (1LL << i)) v[i] = 1.0;
	return v;
}

ll round(const vdbl& v) {
	ll x = 0;
	for (int i = 0; i < v.size(); ++i) if (v[i] >= 0.5) x |= 1LL << i;
	return x;
}

bool roundCheck(const vdbl& tocheck, const vdbl& correct) {
	if (tocheck.size() != correct.size()) {
		printf("Error tocheck.size() = %lu, correct.size() = %lu\n", tocheck.size(), correct.size());
		return false;
	}
	for (int i = 0; i < tocheck.size(); ++i) 
		if ( (tocheck[i] < 0.5 && correct[i] >= 0.5) || (tocheck[i] >= 0.5 && correct[i] < 0.5)) 
			return false;
	return true;
}

ll cat(const ll a, const ll b, int bits) {return (a << bits) | b;}

void Tester::train() {
	Network n(sizes, checker, batchSize, lrate, maxRate, minRate, L2);
	n.SGD(training, testing, numEpochs);
	if ( write ) {
		if (!fout.good() ) cout << "error fout is bad\n";
		else {
			cout << "writing\n";
			fout << n;
		}
	}
}

	