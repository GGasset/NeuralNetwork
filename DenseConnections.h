#include "IConnections.h"
#include "ValueGeneration.h"
#include <string>

#pragma once
class DenseConnections :
    public IConnections
{
protected:
    size_t previous_layer_start_i = -1;

public:
	DenseConnections(size_t previous_layer_start_i, size_t previous_layer_length, size_t neuron_written_gradient_count)
	{
		this->neuron_written_gradient_count = neuron_written_gradient_count;
		weight_count = previous_layer_length;
		GenerateWeights();

		this->previous_layer_start_i = previous_layer_start_i;
	}

	double IConnections::LinearFunction(double* network_activations)
	{
		double linear_function = 0;
		for (size_t i = 0; i < weight_count; i++)
		{
			linear_function += network_activations[previous_layer_start_i + i] * weights[i];
		}
		return linear_function;
	}

	void IConnections::CalculateGradients(double* gradients, double* neuron_activations, double* costs, double linear_function_gradient)
	{
		for (size_t i = 0; i < weight_count; i++)
		{
			gradients[self_gradients_start_i + neuron_written_gradient_count + i] = linear_function_gradient * neuron_activations[previous_layer_start_i + i];
			costs[previous_layer_start_i + i] -= linear_function_gradient * weights[i];
		}
	}

	void IConnections::CalculateGradients(double* gradients, double* neuron_activations, double* costs, double* linear_function_gradients, size_t t_count)
	{
		for (size_t t = 0; t < t_count; t++)
		{
			double current_linear_function_gradient = linear_function_gradients[t];

			size_t first_neuron_gradients_start_i = network_execution_results_value_count * t;
			size_t current_t_first_network_neuron_i = t * network_neuron_count;
			for (size_t i = 0; i < weight_count; i++)
			{
				size_t gradient_i = first_neuron_gradients_start_i + previous_layer_start_i + i + neuron_written_gradient_count;
				size_t connected_activation_i = current_t_first_network_neuron_i + previous_layer_start_i;

				gradients[gradient_i] = current_linear_function_gradient * neuron_activations[connected_activation_i];
				costs[connected_activation_i] -= current_linear_function_gradient * weights[i];
			}
		}
	}

	void IConnections::SubtractGradients(double* gradients)
	{
		for (size_t i = 0; i < weight_count; i++)
		{
			weights[i] -= gradients[previous_layer_start_i + i];
		}
	}

	void IConnections::SubtractGradients(double* gradients, size_t t_count)
	{
		for (size_t t = 0; t < t_count; t++)
		{
			size_t first_neuron_gradients_start_i = network_execution_results_value_count * t;
			for (size_t i = 0; i < weight_count; i++)
			{
				weights[i] -= gradients[first_neuron_gradients_start_i + self_gradients_start_i + neuron_written_gradient_count + i];
			}
		}
	}
};

