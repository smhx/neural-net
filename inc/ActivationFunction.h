#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <cmath>
#include "types.h"

struct SigmoidActivationFunction {
	static Mat activation(const Mat& a);
	static Mat activationDeriv(const Mat& x);
};

struct TanhActivationFunction {
	static Mat activation(const Mat& a);
	static Mat activationDeriv(const Mat& x);
};

struct SoftMaxActivationFunction {
	static Mat activation(const Mat& a);
	static Mat activationDeriv(const Mat& x);
};

#endif