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

vdbl strdiv(const string& str, int namelen) {

	vdbl x(namelen);
	for (int i = 0; i < str.size(); ++i) {
		char c = str[i];
		c = tolower(c);
		x[i] = static_cast<double>(c-'a' + 1) / 26.0;
	}
	for (int i = str.size(); i < namelen; ++i) {
		x[i] = 0.0; // 0.0 for space
	}
	cout << "name = " << str << " and x = ";
	for (auto d : x) cout << d << " ";
	cout << "\n";
	return x;
}

vdbl rstrdiv(const string& str, int namelen) {
	vdbl x(namelen);
	for (int i = 0; i < namelen-str.size(); ++i) x[i] = 0.0;
	for (int i = 0; i < str.size(); ++i) {
		char c = str[i];
		c = tolower(c);
		x[namelen-str.size()+i] = static_cast<double>(c-'a' + 1) / 26.0;
	}
	return x;
}

bool largestCheck(const vdbl& tocheck, const vdbl& correct) {
	int maxarg = 0;
	for (int i = 1; i < tocheck.size(); ++i) {
		if (tocheck[i] > tocheck[maxarg]) maxarg = i;
	}
	return correct[maxarg] > 0.0;
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
	Network n(sizes, checker, batchSize, lrate, maxRate, minRate, L2, momentum);
	n.SGD(training, testing, numEpochs);
	if ( write ) {
		if (!fout.good() ) cout << "error fout is bad\n";
		else {
			cout << "writing\n";
			fout << n;
		}
	}
}

	