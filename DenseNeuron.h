#include "INeuron.h"
#include "DenseConnections.h"

#pragma once
class DenseNeuron : public INeuron
{
public:
	ActivationFunctions::ActivationFunction activation_function;

	/// <param name="weight_direction_from_0">
	/// direction -1: exclusively negative |  
	/// direction 0: not exclusive direction | 
	/// direction 1: exclusively positive
	/// </param>
	DenseNeuron(size_t neuron_i, size_t previous_layer_start_i, size_t previous_layer_length, ActivationFunctions::ActivationFunction activation_function, 
		int8_t weight_direction_from_0 = 0)
	{
		this->neuron_i = neuron_i;
		this->activation_function = activation_function;
		neuron_written_gradient_count = 1;
		neuron_written_execution_results_count = 1;
		connections = new DenseConnections(previous_layer_start_i, previous_layer_length, neuron_written_gradient_count, weight_direction_from_0);
	}

	void INeuron::ExecuteStore(double* activations, double* execution_results, size_t t_index = 0)
	{
		size_t t_activation_addition = t_index * connections->network_neuron_count;
		size_t t_execution_results_addition = t_index * connections->network_execution_results_value_count;
		double linear_function = connections->LinearFunction(activations, t_index) + bias;
		execution_results[t_execution_results_addition + self_execution_results_start_i] = linear_function;
		activations[t_activation_addition + neuron_i] = ActivationFunctions::Activate(linear_function, activation_function);
	}
	
	double INeuron::Execute(double* activations, size_t t_index = 0)
	{
		double linear_function = connections->LinearFunction(activations, t_index) + bias;
		double activation = ActivationFunctions::Activate(linear_function, activation_function);
		activations[connections->network_neuron_count * t_index + neuron_i] = activation;
		return activation;
	}
		
	void INeuron::GetGradients(double* gradients, double* costs, double* execution_results, double* network_activations, size_t t_count)
	{
		double* linear_function_gradients = new double[t_count];
		for (size_t t = 0; t < t_count; t++)
		{
			double current_cost = costs[connections->network_neuron_count * t + neuron_i];
			size_t current_gradient_start_i = connections->network_gradients_value_count * t + self_gradients_start_i;
			size_t current_execution_results_start_i = connections->network_execution_results_value_count * t + self_execution_results_start_i;
			
			linear_function_gradients[t] =
				gradients[current_gradient_start_i] =
					Derivatives::DerivativeOf(execution_results[current_execution_results_start_i], activation_function) * current_cost;
		}

		connections->CalculateGradients(gradients, network_activations, costs, linear_function_gradients, t_count);
		delete[] linear_function_gradients;
	}
	
	void INeuron::SubtractGradients(double* gradients, double learning_rate, size_t t_count)
	{
		for (size_t t = 0; t < t_count; t++)
		{
			size_t gradients_start_i = connections->network_gradients_value_count * t + self_gradients_start_i;
			bias -= gradients[gradients_start_i] * learning_rate;
		}
		connections->SubtractGradients(gradients, t_count, learning_rate);
	}

	void INeuron::DeleteMemory()
	{

	}

	void Free()
	{
		connections->Free();
	}
};

