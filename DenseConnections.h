#include "IConnections.h"
#include "ValueGeneration.h"
#include <string>

#pragma once
class DenseConnections :
    public IConnections
{
protected:
    size_t previous_layer_start_i;

public:
	/// <param name="weight_direction_from_0">
	/// direction -1: exclusively negative |  
	/// direction 0: not exclusive direction | 
	/// direction 1: exclusively positive
	/// </param>
	DenseConnections(size_t previous_layer_start_i, size_t previous_layer_length, size_t neuron_written_gradient_count, int8_t weight_direction_from_0 = 0)
	{
		this->neuron_written_gradient_count = neuron_written_gradient_count;
		weight_count = previous_layer_length;
		GenerateWeights(weight_direction_from_0);

		this->previous_layer_start_i = previous_layer_start_i;
	}

	double IConnections::LinearFunction(double* network_activations, size_t t_index = 0)
	{
		size_t t_addition = t_index * network_neuron_count;
		double linear_function = 0;
		for (size_t i = 0; i < weight_count; i++)
		{
			linear_function += network_activations[t_addition + previous_layer_start_i + i] * weights[i];
		}
		return linear_function;
	}

	double IConnections::CalculateDerivative(double* network_activations, size_t t_index)
	{
		size_t t_addition = t_index * network_neuron_count;
		size_t current_previous_layer_start = t_addition + previous_layer_start_i;

		double derivative = 0;
		for (size_t i = 0; i < weight_count; i++)
		{
			derivative += network_activations[current_previous_layer_start + i] + weights[i];
		}
		return derivative;
	}


	void IConnections::CalculateGradients(double* gradients, double* neuron_activations, double* costs, double* linear_function_gradients, size_t t_count)
	{
		for (size_t t = 0; t < t_count; t++)
		{
			double current_linear_function_gradient = linear_function_gradients[t];

			size_t first_neuron_gradients_start_i = network_gradients_value_count * t;
			size_t current_t_first_network_neuron_i = t * network_neuron_count;
			for (size_t i = 0; i < weight_count; i++)
			{
				size_t gradient_i = first_neuron_gradients_start_i + self_gradients_start_i + neuron_written_gradient_count + i;
				size_t connected_activation_i = current_t_first_network_neuron_i + previous_layer_start_i + i;

				gradients[gradient_i] = current_linear_function_gradient * neuron_activations[connected_activation_i];
				costs[connected_activation_i] -= current_linear_function_gradient * weights[i];
			}
		}
	}

	void IConnections::SubtractGradients(double* gradients, size_t t_count, double learning_rate)
	{
		for (size_t t = 0; t < t_count; t++)
		{
			size_t first_neuron_gradients_start_i = network_gradients_value_count * t;
			for (size_t i = 0; i < weight_count; i++)
			{
				weights[i] -= gradients[first_neuron_gradients_start_i + self_gradients_start_i + neuron_written_gradient_count + i] * learning_rate;
			}
		}
	}
};

