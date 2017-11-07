#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <functional>

typedef long long ll;

typedef std::vector<double> vdbl;
typedef std::vector<vdbl> v2dbl;
typedef std::vector<v2dbl> v3dbl;
typedef std::pair<vdbl, vdbl> trdata; // training data
typedef std::vector<trdata> trbatch;

typedef std::function<bool (const vdbl&, const vdbl&) > checker_type;
#endif