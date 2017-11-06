#include <iostream>
#include <string>

#include "../inc/Network.h"

using namespace std;

typedef vector<double> vdbl;

const int bits = 8;

int read(const vdbl& v) {
	int toret = 0;
	for (int i = 0; i < v.size(); ++i) {
		if (v[i] >= 0.5) toret |= 1 << i;
	}
	return toret;
}


vdbl binary(long long i)
{
	vdbl v(bits, 0.0);
	for (int j = 0; j < bits; ++j) {
		if (i&(1LL << j))
			v[j] = 1.0;
	}
	return v;
}

int main() {
	// string fname;
	// cin >> fname;

	// replace with your paths
	Network n("/Users/Steven/Desktop/neural-net/tests/test.txt");

	while (1) {
		int x;
		cin >> x;
		if (x==-1) break;
		
		vdbl in = binary(x);
		n.feedForward(in);
		int y = read(in);
		cout << "got " << y << " expected " << x*x << "\n";
	}
}