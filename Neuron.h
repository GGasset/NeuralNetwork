#include "INeuron.h"
# include "ActivationFunctions.h"

#pragma once
class Neuron : public INeuron
{
	ActivationFunctions::ActivationFunction activation_function;

	Neuron(IConnections* connections, double bias, ActivationFunctions::ActivationFunction activation_function)
	{
		this->connections = connections;
		this->bias = bias;
		this->activation_function = activation_function;
	}

	double* INeuron::ExecuteStore(double** networkActivations)
	{
		double* output = new double[3];
		double linear_function = connections->LinearFunction(networkActivations);
		output[0] = linear_function;

		linear_function += bias;
		output[1] = linear_function;

		output[2] = ActivationFunctions::Activate(linear_function, this->activation_function);
		return output;
	}
};

