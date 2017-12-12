#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <cmath>

struct SigmoidActivationFunction {
	static double activation(double x);
	static double activationDeriv(double x);
};

struct TanhActivationFunction {
	static double activation(double x);
	static double activationDeriv(double x);
};

#endif