#include <cmath>

#pragma once
class Cost
{
public:
	enum CostFunction
	{
		SquaredMean,
	};

	static double GetCostOf(double neuron_output, double Y, CostFunction cost_function)
	{
		switch (cost_function)
		{
		case Cost::SquaredMean:
			return SquaredMeanLoss(neuron_output, Y);
		default:
			throw std::exception("Cost function not implemented");
		}
	}

	static double SquaredMeanLoss(double neuronOutput, double Y)
	{
		return pow(neuronOutput - Y, 2);
	}
};

