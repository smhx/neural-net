#include "../tester.h"
#include <fstream>
#include <algorithm>
using namespace std;

const int namelen = 10;

// const int trsize = 2800;

const int tstsize = 100;

// There are 2943 males
// There are 5004 females

void train() {
	srand(time(NULL));
	Tester tester;

	tester.sizes = {namelen, 10*namelen, 2}; // output is (0.0, 1.0) for female (1.0, 0.0) for male

	ifstream guys("test/gender/males.txt"), girls("test/gender/females.txt");

	vector<string> guynames, girlnames;

	string name;
	while (guys >> name) if (name.size() < namelen) guynames.push_back(name);
	while (girls >> name) if (name.size() < namelen) girlnames.push_back(name);


	for (string n : guynames) {
		tester.training.push_back({strdiv(name, namelen), {1.0, 0.0}});
	}

	for (string n : girlnames) {
		tester.training.push_back({strdiv(name, namelen), {0.0, 1.0}});
	}


	for (int i = 0; i < tstsize; ++i) {
		int x = rand() % guynames.size();
		tester.testing.push_back({strdiv(guynames[x], namelen), {1.0, 0.0}});
		x = rand() % girlnames.size();
		tester.testing.push_back({strdiv(girlnames[x], namelen), {0.0, 1.0}});
	}

	printf("Finished initializing\n");

	tester.numEpochs = 200;
	tester.batchSize = 20;

	tester.lrate = 100.0;
	tester.maxRate = 2.0;
	tester.minRate = 1.0;
	
	tester.checker = largestCheck;

	tester.write = true;

	printf("Training\n");

	tester.fout = ofstream("test/gender/net.txt");

	tester.train();

}

void test() {

	printf("Testing\n");

	ifstream fin("test/gender/net.txt");

	Network n(fin, largestCheck);

	while (1) {
		string name; 
		cin >> name;
		if (name.size() >= namelen) continue;

		vdbl in = strdiv(name, namelen);
		n.feedForward(in);

		cout << "got (" << in[0] << ", " << in[1] << ".\n";
	}
}

int main() {
	train();
	test();
}