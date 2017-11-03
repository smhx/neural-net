#ifndef DATA_H
#define DATA_H

struct Data {
	int in, out;
	Data(int _in, int _out);
	Data(const Data& data);
	double outdbl(); // returns result as double
};

#endif