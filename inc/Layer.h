#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>
#include "types.h"

// The layer interface. Is an abstract class

class Layer {
public:

	Layer();
	virtual ~Layer();

	virtual void apply(Mat& input)=0;

	// if this layer is the last layer, computes the delta (error) given the output and correct answer
	virtual void computeDeltaLast(const Mat& output, const Mat& ans, Mat& WTD)=0;

	// if this layer is not the last layer, computes the delta from the last layer's delta
	virtual void computeDeltaBack(Mat& WTD)=0;

	virtual void updateBiasAndWeights(double lrate)=0;

	virtual std::pair<int, int> getSize()=0;

	virtual void print()=0;
};

#endif