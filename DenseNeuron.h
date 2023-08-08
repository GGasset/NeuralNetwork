#include "INeuron.h"
#include "DenseConnections.h"
#include "ActivationFunctions.h"
#include "Derivatives.h"

#pragma once
class DenseNeuron : public INeuron
{
public:
	ActivationFunctions::ActivationFunction activation_function;

	DenseNeuron(size_t neuron_i, size_t previous_layer_start_i, size_t previous_layer_length, ActivationFunctions::ActivationFunction activation_function)
	{
		this->neuron_i = neuron_i;
		this->activation_function = activation_function;
		neuron_written_gradient_count = 1;
		connections = new DenseConnections(previous_layer_start_i, previous_layer_length, neuron_written_gradient_count);
	}

	void INeuron::ExecuteStore(double* activations, double* execution_results, size_t t_index = 0)
	{
		size_t t_activation_adition = t_index * connections->network_neuron_count;
		size_t t_execution_results_adition = t_index * connections->network_execution_results_value_count;
		double linear_function = connections->LinearFunction(activations, t_index) + bias;
		execution_results[t_execution_results_adition + self_execution_results_start_i] = linear_function;
		activations[t_activation_adition + neuron_i] = ActivationFunctions::Activate(linear_function, activation_function);
	}
	
	double INeuron::Execute(double* activations, size_t t_index = 0)
	{
		double linear_function = connections->LinearFunction(activations, t_index) + bias;
		double activation = ActivationFunctions::Activate(linear_function, activation_function);
		activations[connections->network_neuron_count * t_index + neuron_i] = activation;
		return activation;
	}
	
	//void INeuron::GetGradients(double* gradients, double* costs, double* execution_results, double* network_activations)
	//{
	//	double current_cost = costs[neuron_i];
	//	double linear_function = execution_results[self_execution_results_start_i];
	//	double linear_function_gradient =
	//		gradients[self_execution_results_start_i] = current_cost * Derivatives::DerivativeOf(linear_function, activation_function);
	//	connections->CalculateGradients(gradients, network_activations, costs, linear_function_gradient);
	//}
	
	void INeuron::GetGradients(double* gradients, double* costs, double* execution_results, double* network_activations, size_t t_count)
	{
		double* linear_function_gradients = new double[t_count];
		for (size_t t = 0; t < t_count; t++)
		{
			double current_cost = costs[connections->network_neuron_count * t + neuron_i];
			size_t current_gradient_start_i = connections->network_execution_results_value_count * t + self_execution_results_start_i;
			
			linear_function_gradients[t] =
				gradients[current_gradient_start_i] =
					Derivatives::DerivativeOf(execution_results[current_gradient_start_i], activation_function) * current_cost;
		}

		connections->CalculateGradients(gradients, network_activations, costs, linear_function_gradients, t_count);
		delete[] linear_function_gradients;
	}
	
	void INeuron::SubtractGradients(double* gradients, double learning_rate)
	{
		bias -= gradients[self_execution_results_start_i] * learning_rate;
		connections->SubtractGradients(gradients, learning_rate);
	}
	
	void INeuron::SubtractGradients(double* gradients, double learning_rate, size_t t_count)
	{
		for (size_t t = 0; t < t_count; t++)
		{
			size_t gradients_start_i = connections->network_execution_results_value_count * t + self_execution_results_start_i;
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

