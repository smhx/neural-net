#include "../inc/Network.h"

using namespace std; // does not affect main.cpp
/*

// Initialize the parameters and sets random weights and biases
Network::Network(const vector<int>& sizes, const checker_type& c, int _batchSize, double _learnRate, double _maxLearn, double _minLearn, double _L2weight, double _momentum) {

	checker = c;
	batchSize = _batchSize;
	learnRate = _learnRate;
	maxLearn = _maxLearn;
	minLearn = _minLearn;
	L2weight = _L2weight;
	momentum = _momentum;
	randGen = mt19937(randDev()); 

	// defaults to mean of 0.0, standard dev of 1.0
	randDistribution = normal_distribution<double>(); 

	numLayers = sizes.size();

	layerSizes = sizes;

	biases = v2dbl(sizes.size());
	weights = velocity = v3dbl(sizes.size() - 1);

	for (int i = 0; i < numLayers; ++i) {
		// don't set for input layer
		if (i) { 
			biases[i] = vdbl(sizes[i]);
			for (int j = 0; j < sizes[i]; ++j) {
				biases[i][j] = randDistribution(randGen); // is a random double
			}
		}
		// don't set for output layer
		if (i < numLayers - 1)
		{
			weights[i] = velocity[i] = v2dbl(sizes[i]);
			for (int j = 0; j < sizes[i]; ++j)	{
				weights[i][j] = velocity[i][j] = vdbl(sizes[i + 1]);
				for (int k = 0; k < sizes[i+1]; ++k) {
					weights[i][j][k] = randDistribution(randGen) / sqrt(sizes[i]);
				}
			}
		}
	}
}

// Initializes the parameters, weights, and biases from file
Network::Network(ifstream& fin, const checker_type& c) {
	checker = c;
	if (!fin.good()) {
		printf("Error in Network file construtor\n");
		return;
	}
	fin >> numLayers;
	cout << "Number of Layers = " << numLayers << "\n";
	layerSizes = vector<int>(numLayers);
	for (int i = 0; i < numLayers; ++i) {
		fin >> layerSizes[i];
	}
	printf("Layer Sizes:");
	for (int i = 0; i < numLayers; ++i) {
		printf(" %d%c", layerSizes[i], (i==numLayers-1 ? '\n' : ','));
	}
	fin >> batchSize >> learnRate >> maxLearn >> minLearn >> L2weight >> momentum;
	printf("Batch Size = %d\n", batchSize);
	printf("Learning Rate: start = %lf, max = %lf, min = %lf\n", learnRate, maxLearn, minLearn);
	printf("L2 Weight = %lf\n", L2weight);
	printf("Momentum Coefficient = %lf\n", momentum);
	biases = v2dbl(numLayers);
	for (int i = 1; i < numLayers; ++i) {
		biases[i] = vdbl(layerSizes[i]);
		for (int j = 0; j < layerSizes[i]; ++j) {
			fin >> biases[i][j];
		}
	}
	printf("Got biases\n");
	weights = v3dbl(numLayers - 1);
	for (int i = 0; i + 1 < numLayers; ++i) {
		weights[i] = v2dbl(layerSizes[i]);
		for (int j = 0; j < layerSizes[i]; ++j) {
			weights[i][j] = vdbl(layerSizes[i+1]);
			for (int k = 0; k < layerSizes[i+1]; ++k){
				fin >> weights[i][j][k];
			}
		}
	}
	printf("Got weights\n");
	velocity = v3dbl(numLayers - 1);
	for (int i = 0; i + 1 < numLayers; ++i)	{
		velocity[i] = v2dbl(layerSizes[i]);
		for (int j = 0; j < layerSizes[i]; ++j)	{
			velocity[i][j] = vdbl(layerSizes[i + 1]);
			for (int k = 0; k < layerSizes[i + 1]; ++k)	{
				fin >> velocity[i][j][k];
			}
		}
	}
	printf("Got velocities\n");
}

Network& Network::operator=(const Network& net) {
	checker = net.checker;
	batchSize = net.batchSize;
	learnRate = net.learnRate;
	maxLearn = net.maxLearn;
	minLearn = net.minLearn;
	L2weight = net.L2weight;
	momentum = net.momentum;
	layerSizes = net.layerSizes;
	weights = net.weights;
	biases = net.biases;
	velocity = net.velocity;

	randGen = mt19937(randDev()); 

	randDistribution = normal_distribution<double>(); 
	return *this;
}


ofstream& operator<<(ofstream& fout, const Network& n) {
	fout << n.numLayers << "\n";
	for (int sz : n.layerSizes) fout << sz << " ";
	fout << "\n";
	fout << n.batchSize << " " << n.learnRate << " " << n.maxLearn << " " << n.minLearn << " " << n.L2weight << " " << n.momentum << "\n";
	for (int i = 1; i < n.numLayers; ++i) {
		for (int j = 0; j < n.layerSizes[i]; ++j) {
			fout << n.biases[i][j] << " ";
		}
		fout << "\n";
	}
	for (int i = 0; i + 1 < n.numLayers; ++i) {
		for (int j = 0; j < n.layerSizes[i]; ++j) {
			for (int k = 0; k < n.layerSizes[i+1]; ++k){
				fout << n.weights[i][j][k] << " ";
			}
		}
		fout << "\n";
	}
	for (int i = 0; i + 1 < n.numLayers; ++i) {
		for (int j = 0; j < n.layerSizes[i]; ++j) {
			for (int k = 0; k < n.layerSizes[i+1]; ++k){
				fout << n.velocity[i][j][k] << " ";
			}
		}
		fout << "\n";
	}
	return fout;
}

void Network::feedForward(vdbl& a) {
	// a is initially the vector of inputs
	vdbl dot;
	for (int i = 0; i < numLayers - 1; ++i) {
		// update layer i + 1 from layer i
		dot = vdbl(layerSizes[i+1], 0.0);
		for (int j = 0; j < layerSizes[i]; ++j) {
			for (int k = 0; k < layerSizes[i+1]; ++k) {
				dot[k] += weights[i][j][k] * a[j];
			}
		}
		a.resize(dot.size());
		for (int k = 0; k < a.size(); ++k) {
			a[k] = sigmoid(dot[k]+biases[i+1][k]);
		}
	}
}

void Network::SGD(trbatch& data, trbatch& test, int numEpochs, string fname) {
	for (int epoch = 1; epoch <= numEpochs; ++epoch) {
		shuffle(data.begin(), data.end(), randGen);
		vector< trbatch > batches;
		for (int i = 0; i < data.size(); ++i) {
			if (i%batchSize == 0) batches.push_back(trbatch());
			batches.back().push_back(data[i]);
		}
		for (auto batch : batches) {
			updateBatch(batch);
		}
		printf("Epoch %d: ", epoch);
		testBatch(test);
		ofstream fout(fname);
		fout << *this;
	}
}

void Network::updateBatch(const trbatch& batch) {

	// gradb[i][j] is the gradient for the bias of the jth node in layer i
	// gradw[i][j][k] is the gradient for the weight from the jth node in layer i to the kth node in layer i+1
	v2dbl gradb(numLayers), dgradb(numLayers);
	for (int i = 0; i < numLayers; ++i) {
		gradb[i] = dgradb[i] = vdbl(layerSizes[i], 0.0);
	}
	v3dbl gradw(numLayers - 1), dgradw(numLayers - 1);
	for (int i = 0; i < numLayers - 1; ++i)	{
		gradw[i] = dgradw[i] = v2dbl(layerSizes[i]);
		for (int j = 0; j < layerSizes[i]; ++j)	{
			gradw[i][j] = dgradw[i][j] = vdbl(layerSizes[i + 1], 0.0);
		}
	}
	// Range based loops by constant reference!
	for (const trdata& data : batch) { 

		// dgradb[i][j] is the partial derivative of the cost function 
		// for the current data relative to b[i][j]
		// dgradw[i][j][k] is the partial derivative of the cost function
		// for the current data relative to w[i][j][k]

		// pass dgradb and dgradw by reference
		backprop(data, dgradb, dgradw); 


		for (int i = 0; i < gradb.size(); ++i) {
			for (int j = 0; j < gradb[i].size(); ++j) {
				gradb[i][j] += dgradb[i][j];
			}
		}
		for (int i = 0; i < gradw.size(); ++i) {
			for (int j = 0; j < gradw[i].size(); ++j) {
				for (int k = 0; k < gradw[i][j].size(); ++k) {
					gradw[i][j][k] += dgradw[i][j][k];
				}
			}
		}
	}
	double n = static_cast<double>(batch.size());
	for (int i = 0; i < biases.size(); ++i) {
		for (int j = 0; j < biases[i].size(); ++j) {
			biases[i][j] -= learnRate/n * gradb[i][j];
		}
	}

	for (int i = 0; i < numLayers-1; ++i) {
		for (int j = 0; j < layerSizes[i]; ++j) {
			for (int k = 0; k < layerSizes[i+1]; ++k) {
				velocity[i][j][k] = momentum*velocity[i][j][k] - learnRate / n * gradw[i][j][k];
				weights[i][j][k] = (1 - learnRate*L2weight / n)*weights[i][j][k] + velocity[i][j][k];
			}
		}
	}

}

void Network::backprop(const trdata& data, v2dbl& dgradb, v3dbl& dgradw)
{
	vdbl a = data.first;
	v2dbl activations(numLayers);
	activations[0] = a;
	v2dbl zs(numLayers);

	// feedforward
	for (int i = 0; i < numLayers - 1; ++i)
	{
		// calculate a and z of layer i+1 from layer i
		// a = sigma(z)
		vdbl z = biases[i + 1];
		for (int j = 0; j < layerSizes[i]; ++j) {
			for (int k = 0; k < layerSizes[i + 1]; ++k)	{
				z[k] += weights[i][j][k] * a[j];
			}
		}
		zs[i + 1] = z;
		a.resize(layerSizes[i + 1]);
		a = sigmoid(z);
		activations[i + 1] = a;
	}

	// backpropagate
	vdbl delta(layerSizes[numLayers - 1]);
	for (int i = numLayers - 1; i > 0; --i) {
		if (i == numLayers - 1)	{
			// calculate delta of last layer
			vdbl sp = sigmoidPrime(zs[i]);
			vdbl x = costDerivative(activations[i], data.second);
			// delta = multiply(x, sp); //for quadratic cost
			delta = x; // for cross-entropy cost
		}
		else {
			// calculate delta of layer i from layer i+1
			vdbl sp = sigmoidPrime(zs[i]);
			vdbl x(layerSizes[i], 0.0);
			for (int j = 0; j < layerSizes[i]; ++j)	{
				for (int k = 0; k < layerSizes[i + 1]; ++k)	{
					x[j] += weights[i][j][k] * delta[k];
				}
			}
			delta = multiply(x, sp);
		}

		// calculate change for bias[i] and weights[i-1]
		dgradb[i] = delta;
		for (int j = 0; j < layerSizes[i-1]; ++j) {
			for (int k = 0; k < layerSizes[i]; ++k)	{
				dgradw[i-1][j][k] = activations[i-1][j] * delta[k];
			}
		}
	}
	return;
}

void Network::testBatch(const trbatch& batch) {
	int count = 0;
	double cost = 0.0;
	for (trdata data : batch) {
		vdbl in = data.first, out = data.second;
		feedForward(in);
		
		if (checker(in, out)) ++count;
		//calculate cost
		for (int i = 0; i < in.size(); ++i) {
			// cost += 0.5*(in[i] - out[i])*(in[i] - out[i]) / (batch.size()); // for quadratic cost
			if (in[i] <= 0 || 1-in[i]<= 0) {
				// printf("ERROR!!!!!!! in[i] = %lf, out[i] = %lf\n", in[i], out[i]);
				continue;
			}

			cost -= (out[i] * log(in[i]) + (1 - out[i])*log(1 - in[i])) / batch.size();
		}
		for (int i = 0; i < weights.size(); ++i) {
			for (int j = 0; j < weights[i].size(); ++j)	{
				for (int k = 0; k < weights[i][j].size(); ++k) {
					cost += L2weight * 0.5 / batch.size() * weights[i][j][k] * weights[i][j][k];
				}
			}
		}
	}
	double frac = (double)count / (double)batch.size();
	maxfrac = max(maxfrac, frac);
	learnRate = 0.3*learnRate + 0.7*(frac*minLearn + (1-frac)*maxLearn);

	double maxWeight = 0;
	// find max weight
	for (int i = 0; i < numLayers - 1; ++i)
		for (int j = 0; j < layerSizes[i]; ++j)
			for (int k = 0; k < layerSizes[i + 1]; ++k)
				maxWeight = max(maxWeight, abs(weights[i][j][k]));

	printf("%d/%lu, cost=%.3lf, lrate=%lf, mf=%.2lf, mw=%.1f\n", count, batch.size(), cost, learnRate, maxfrac, maxWeight);
}

inline double Network::sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

inline double Network::sigmoidPrime(double x) { return sigmoid(x)*(1 - sigmoid(x)); }

inline vdbl Network::sigmoid(const vdbl& x) {
	vdbl y(x.size());
	for (int i = 0; i < x.size(); ++i)
		y[i] = sigmoid(x[i]);
	return y;
}

inline vdbl Network::sigmoidPrime(const vdbl& x) {
	vdbl y(x.size());
	for (int i = 0; i < x.size(); ++i)
		y[i] = sigmoidPrime(x[i]);
	return y;
}

inline vdbl Network::multiply(const vdbl& x, const vdbl& y) {
	vdbl z(x.size());
	for (int i = 0; i < x.size(); ++i)
		z[i] = x[i]*y[i];
	return z;
}

inline vdbl Network::costDerivative(const vdbl& activation, const vdbl& ans) {
	if (activation.size() != ans.size())
		printf("Error in costDerivative: activation.size = %lu, ans.size = %lu\n", activation.size(), ans.size());
	vdbl x(activation.size());
	for (int i = 0; i < activation.size(); ++i)
		x[i] = activation[i] - ans[i];
	return x;
}

*/