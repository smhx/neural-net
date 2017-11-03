#ifndef DATA_H
#define DATA_H

#include <vector>

struct Data {
	typedef std::vector<double> vdbl;

	int in, out;
	vdbl inlayer, outlayer;

	Data(int _in, int _out);
	// Data(int _in, const vdbl& _outlayer);
};

#endif