// this file just for random tests
#include "inc/ActivationFunction.h"
#include "inc/types.h"
#include <Eigen/Dense>
#include <iostream>

int main() {
	Mat x(3,3);
	x << 1, 2, 3,
	     4, 5, 6,
	     7, 8, 9;
	std::cout << "x = \n" << x << std::endl;
	Mat y = SoftMaxActivationFunction::activation(x);
	std::cout << y << std::endl;
}
