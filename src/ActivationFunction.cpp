#include "../inc/ActivationFunction.h"
#include <cmath>

static double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

static double sigmoidDeriv(double x) {
	return sigmoid(x)*(1-sigmoid(x));
}


static double tanhDeriv(double x) {
	return 1 - (tanh(x)*tanh(x));	
}

Mat SigmoidActivationFunction::activation(const Mat& x) {
	return x.unaryExpr(&sigmoid);
}
Mat SigmoidActivationFunction::activationDeriv(const Mat& x) {
	return x.unaryExpr(&sigmoidDeriv);
}

Mat TanhActivationFunction::activation(const Mat& x) {
	return x.unaryExpr(&tanh);
}
Mat TanhActivationFunction::activationDeriv(const Mat& x) {
	return x.unaryExpr(&tanhDeriv);
}

Mat SoftMaxActivationFunction::activation(const Mat& x) {

	// Probs a better way to do it.
	Mat y = x.unaryExpr(&exp);
	for (int c = 0; c < y.cols(); ++c){
		double sum = y.col(c).sum();
		y.col(c) /= sum;
	}
	return y;
}