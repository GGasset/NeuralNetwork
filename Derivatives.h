#include "Cost.h"

#pragma once
static class Derivatives
{
public:
	// Activation Functions derivatives

	static double DerivativeOf(double x, ActivationFunctions::ActivationFunction ActivationType)
	{
		switch (ActivationType)
		{
		case ActivationFunctions::RELU:
			return RELUDerivative(x);
		case ActivationFunctions::Sigmoid:
			return SigmoidDerivative(x);
		case ActivationFunctions::Tanh:
			return TanhDerivative(x);
		case ActivationFunctions::None:
			return 1;
		default:
			return NULL;
		}
	}

	static double RELUDerivative(double x)
	{
		return x * (x >= 0);
	}

	static double SigmoidDerivative(double x)
	{
		return ActivationFunctions::SigmoidActivation(x) * (1 - ActivationFunctions::SigmoidActivation(x));
	}

	static double TanhDerivative(double x)
	{
		double bottomDerivative = expDerivative(-2 * x);
		return -DivisionDerivative(2, 1 + exp(-2 * x), 0, bottomDerivative);
	}

	static double DivisionDerivative(double a, double b, double Da, double Db)
	{
		return (Da * b - Db * a) / (b * b);
	}

	static double expDerivative(double x)
	{
		return exp(x);
	}

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

