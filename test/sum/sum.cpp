#include "../tester.h"
#include "../../inc/Network.h"

using namespace std;

const int bits = 5;

int main() {
	srand(time(NULL));
	Tester tester;

	tester.sizes = {2*bits, 10*bits, bits+1};

	for (ll i = 0; i < 1LL << bits; ++i) {
		for (ll j = 0; j < 1LL <<bits; ++j) {
			tester.training.push_back({tobinary(cat(i, j, bits), 2*bits), tobinary(i+j, bits+1) });
		}
	}


	for (int i = 0; i < 100; ++i) {
		ll x = rand() & ( (1 << bits) -1);
		ll y = rand() & ( (1 << bits) -1);
		tester.testing.push_back({tobinary(cat(x, y, bits), 2*bits), tobinary(x+y, bits+1)});
	}

	printf("Finished initializing\n");
	
	tester.numEpochs = 10;
	tester.batchSize = 20;

	tester.lrate = 2.0;
	tester.maxRate = 2.0;
	tester.minRate = 1.0;
	
	tester.checker = roundCheck;

	tester.write = true;

	printf("Training\n");

	tester.fout = ofstream("test/sum/sum.txt");

	tester.train();

}