#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <functional>
#include <Eigen/Dense>

typedef long long ll;

typedef std::vector<double> vdbl;
typedef std::vector<vdbl> v2dbl;
typedef std::vector<v2dbl> v3dbl;

typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;

//typedef std::pair<vdbl, vdbl> trdata; // training data
typedef std::pair<Vec, Vec> trdata;
typedef std::vector<trdata> trbatch;

typedef std::function<std::pair<int, double> (const Mat&, const Mat&) > checker_type;
#endif