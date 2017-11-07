#include "../inc/Network.h"
#include "../inc/types.h"

#include <string>
#include <iostream>

vdbl tobinary(ll x, int bits);

// returns a vdbl of size namelen with  each entry 
// x[i] = (tolower(str[i])-'a')/26.0 if str[i] is an alpha
// if i >= str.size() it's just set to 1.0
vdbl strdiv(const std::string& str, int namelen);

ll round(const vdbl& v);

// assumes correct has exactly 1 1.0, rest are 0.0. Returns whether maximum of tocheck 
// has same index of the 1 in correct
bool largestCheck(const vdbl& tocheck, const vdbl& correct); 

// returns whether the rounded tocheck and correct are exactly the sames
bool roundCheck(const vdbl& tocheck, const vdbl& correct); // does NOT assume correct is either 0 or 1

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