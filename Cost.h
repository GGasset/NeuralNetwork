#include <cmath>

#pragma once
class Cost
{
public:
	enum CostFunction
	{
		SquaredMean,
	};

	static float GetCostOf(size_t networkOutputLength, double* neuronOutput, double* Y, CostFunction costType)
	{
		switch (costType)
		{
		case Cost::SquaredMean:
			return SquaredMeanLoss(networkOutputLength, neuronOutput, Y);
		default:
			return NULL;
		}
	}

	static double SquaredMeanLoss(size_t outputLength, double* neuronOutput, double* Y)
	{
		float mean = 0;
		for (size_t i = 0; i < outputLength; i++)
		{
			mean += SquaredMeanLoss(neuronOutput[i], Y[i]);
		}
	}

	static float SquaredMeanLoss(double neuronOutput, double Y)
	{
		return pow(neuronOutput - Y, 2);
	}
};

