// nothing lol
#include "../inc/Data.h"

Data::Data(int _in, int _out) : in(_in), out(_out) {}

Data::Data(const Data& data) : in(data.in), out(data.out) {}
