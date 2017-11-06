#include <iostream>
#include <string>

#include "../inc/Network.h"

using namespace std;

typedef vector<double> vdbl;

typedef long long ll;

ll read(const vdbl& v) {
	int toret = 0;
	for (int i = 0; i < v.size(); ++i) {
		if (v[i] >= 0.5) toret |= 1LL << i;
	}
	return toret;
}


vdbl binary(long long i, int bits)
{
	vdbl v(bits, 0.0);
	for (int j = 0; j < bits; ++j) {
		if (i&(1LL << j))
			v[j] = 1.0;
	}
	return v;
}


long long cat(ll a, ll b, int bits) {
	return (a << bits) | b;
}

bool check(const vdbl& tocheck, const vdbl& correct) {
	if (tocheck.size() != correct.size()) {
		printf("ERROR different sze\n");
		return false;
	}
	bool works = true;
	for (int i = 0; i < tocheck.size() && works; ++i) {
		if (abs(tocheck[i]-correct[i]) >= 0.5) works = false;
	}
	return works;
}


int main() {
	// string fname;
	// cin >> fname;
	// replace with your paths
	Network n("/Users/Steven/Desktop/neural-net/tests/test.txt", check);

	int bits = 6;
	while (1) {
		ll x, y;
		cin >> x >> y;
		if (x==-1) break;
		
		vdbl in = binary(cat(x, y, bits), 2*bits);
		n.feedForward(in);
		ll z = read(in);
		cout << "got " << z << " expected " <<  x+y << "\n";
	}

	// for (int z = 0; z < 1<<bits; ++z) {
	// 	int x = fib[z];
	// 	vdbl in = binary(z);
	// 	n.feedForward(in);
	// 	int y = read(in);
	// 	if (y!=x) {
	// 		cout << z << " wrong. expected " << x << " got " << y << endl;
	// 	}
	// }
}