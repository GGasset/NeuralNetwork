#include "Cost.h"

#pragma once
class Derivatives
{
public:
	//Cost derivatives

	static double* DerivativeOf(size_t networkOutputLength, double* networkOutput, double* Y, Cost::CostFunction costFunction)
	{
		switch (costFunction)
		{
		case Cost::SquaredMean:
			return SquaredMeanDerivative(networkOutputLength, networkOutput, Y);
		default:
			return NULL;
		}
	}


	static double* SquaredMeanDerivative(size_t networkOutputLength, double* networkOutput, double* Y)
	{
		double* output = new double[networkOutputLength];
		for (size_t i = 0; i < networkOutputLength; i++)
		{
			output[i] = SquaredMeanDerivative(networkOutput[i], Y[i]);
		}
		return output;
	}

	static double SquaredMeanDerivative(double neuronOutput, double Y)
	{
		return 2 * (neuronOutput - Y);
	}
};

