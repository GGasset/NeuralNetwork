#include "INeuron.h"
# include "ActivationFunctions.h"
#include "Derivatives.h"

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
		double* output = new double[2];
		double linear_function = connections->LinearFunction(networkActivations);

		linear_function += bias;
		output[0] = linear_function;

		output[1] = ActivationFunctions::Activate(linear_function, this->activation_function);
		return output;
	}

	double INeuron::GetOutput(double* execute_store_output)
	{
		return execute_store_output[1];
	}

	double INeuron::Execute(double** network_activations)
	{
		double* full_output = ExecuteStore(network_activations);
		double output = GetOutput(full_output);
		free(full_output);
		return output;
	}

	double* INeuron::GetGradients(double* execution_results, double neuron_cost, double** network_costs, double** network_activations)
	{
		double* gradients = new double[1 + connections->GetWeightCount()];

		double linear_function_gradient = neuron_cost * Derivatives::DerivativeOf(execution_results[0], this->activation_function);
		gradients[0] = linear_function_gradient;

		connections->GetGradients(1, gradients, network_activations, network_costs, gradients[0]);

		return gradients;
	}

	void INeuron::GetGradients(double**** execution_results, size_t calculated_steps, double**** output, double*** network_costs, double*** network_activations)
	{
		size_t layer_i = connections->GetLayerI();
		size_t neuron_i = connections->GetNeuronI();
		for (size_t t = 0; t < calculated_steps; t++)
		{
			output[t][layer_i][neuron_i] = GetGradients(execution_results[t][layer_i][neuron_i], network_costs[t][layer_i][neuron_i], network_costs[t], network_activations[t]);
		}
	}

	void INeuron::SubtractGradients(double* neuron_gradients, double learning_rate)
	{
		bias -= neuron_gradients[0];
		connections->SubtractGradients(neuron_gradients, 1, learning_rate);
	}

	void SubtractGradients(double**** network_gradients_over_t, size_t calculated_steps, double learning_rate)
	{
		size_t layer_i = this->connections->GetLayerI();
		size_t neuron_i = this->connections->GetNeuronI();
		for (size_t t = 0; t < calculated_steps; t++)
		{
			SubtractGradients(network_gradients_over_t[t][layer_i][neuron_i], learning_rate);
		}
	}

	void DeleteMemory()
	{

	}

	void Free()
	{
		connections->Free();
	}
};

