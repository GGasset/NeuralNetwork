#include "IConnections.h"
#include "ValueGeneration.h"
#include <string>

#pragma once
class DenseConnections :
    public IConnections
{
protected:
    size_t previous_layer_start_i = -1;
    size_t previous_layer_length = -1;

public:
	DenseConnections(size_t previous_layer_start_i, size_t previous_layer_length, 
		size_t self_gradients_start_i, size_t self_gradients_length, size_t neuron_written_gradient_count)
	{
		this->self_gradients_start_i = self_gradients_start_i;
		this->self_gradients_length = self_gradients_length;
		this->neuron_written_gradient_count = neuron_written_gradient_count;
		weight_count = previous_layer_length;
		GenerateWeights();

		this->previous_layer_start_i = previous_layer_start_i;
		this->previous_layer_length = previous_layer_length;
	}

	double IConnections::LinearFunction(double* network_activations)
	{
		double linear_function = 0;
		for (size_t i = 0; i < previous_layer_length; i++)
		{
			linear_function += network_activations[previous_layer_start_i + i] * weights[i];
		}
		return linear_function;
	}

	void IConnections::CalculateGradients(double* gradients, double* neuron_activations, double* costs, double linear_function_gradient)
	{

	}

	void IConnections::SubtractGradients(double* gradients)
	{

	}
};

