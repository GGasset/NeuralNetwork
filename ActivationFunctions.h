#pragma once
#include <math.h>
#include <cmath>
#include <stdexcept>

class ActivationFunctions
{
public:
	enum ActivationFunction
	{
		None,
		RELU,
		Sigmoid,
		Tanh,
	};

	static double Activate(double x, ActivationFunction activationType)
	{
		switch (activationType)
		{
		case ActivationFunctions::RELU:
			return RELUActivation(x);
		case ActivationFunctions::Sigmoid:
			return SigmoidActivation(x);
		case Tanh:
			return TanhActivation(x);
		case None:
			return x;
		default:
			return NULL;
		}
	}

	static double RELUActivation(double x)
	{
		return fmax(0, x);
	}

	static double SigmoidActivation(double x)
	{
		return 1 / (1 + exp(-x));
	}

	static double TanhActivation(double x)
	{
		return (2 / (1 + exp(-2 * x))) - 1;
	}
};

