#ifndef DATA_H
#define DATA_H

struct Data {
	double in, out;
	Data(int _in, int _out);
	Data(const Data& data);
};

#endif