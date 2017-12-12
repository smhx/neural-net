#include "../inc/ActivationFunction.h"

double SigmoidActivationFunction::activation(double x) {
	return 1.0 / (1.0 + exp(-x));
}
double SigmoidActivationFunction::activationDeriv(double x) {
	return activation(x)*(1 - activation(x));		
}

double TanhActivationFunction::activation(double x) {
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x));	
}
double TanhActivationFunction::activationDeriv(double x) {
	return 1 - (activation(x)*activation(x));	
}