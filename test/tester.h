#include "../inc/Network.h"
#include "../inc/types.h"

#include <string>
#include <iostream>

vdbl tobinary(ll x, int bits);

ll round(const vdbl& v);
bool roundCheck(const vdbl& tocheck, const vdbl& correct);

ll cat(const ll a, const ll b, int bits);


struct Tester {

	void train();

	bool write = false;

	std::ofstream fout;

	trbatch training, testing;

	int batchSize, numEpochs;

	double lrate, maxRate, minRate,  L2;

	checker_type checker;

	std::vector<int> sizes;
	
};