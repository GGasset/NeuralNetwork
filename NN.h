#include <list>
#include <tuple>

#include "INeuron.h"
#include "Derivatives.h"
#include "Cost.h"

#pragma once
class NN
{
private:
	INeuron** neurons = 0;
	size_t neuron_count = -1;
	size_t execution_results_value_count = -1;
	size_t gradients_value_count = -1;
	size_t input_length = -1;
	size_t output_length = -1;

public:
	/// <param name="input_layer_length">This layer is not instantiated as neurons</param>
	NN(INeuron** neurons, size_t neuron_count, size_t input_layer_length, size_t output_layer_length)
	{
		input_length = input_layer_length;
		output_length = output_layer_length;
		this->neuron_count = neuron_count;

		this->neurons = neurons;
		size_t network_execution_results_value_count = 0;
		size_t network_gradients_value_count = 0;
		for (size_t i = 0; i < neuron_count; i++)
		{
			INeuron* current_neuron = neurons[i];

			// Set
			current_neuron->self_execution_results_start_i = network_execution_results_value_count;
			current_neuron->self_gradients_start_i = network_gradients_value_count;

			current_neuron->connections->self_gradients_start_i = network_execution_results_value_count;
			current_neuron->connections->self_gradients_start_i = network_gradients_value_count;
			current_neuron->connections->network_neuron_count = neuron_count + input_length;


			//Get
			network_execution_results_value_count += current_neuron->GetNeuronWrittenExecutionResultsCount();
			network_gradients_value_count += current_neuron->GetNeuronWrittenGradientCount();
			network_gradients_value_count += current_neuron->connections->GetWeightCount();

		}

		for (size_t i = 0; i < neuron_count; i++)
		{
			neurons[i]->connections->network_execution_results_value_count = network_execution_results_value_count;
			neurons[i]->connections->network_gradients_value_count = network_gradients_value_count;
		}

		this->execution_results_value_count = network_execution_results_value_count;
		this->gradients_value_count = network_gradients_value_count;
	}

private:
	void ExecuteStore(double* X, double* network_activations, double* execution_results, size_t t_index = 0)
	{
		for (size_t i = 0; i < input_length; i++)
		{
			network_activations[i + (input_length + neuron_count) * t_index] = X[i + t_index * input_length];
		}
		for (size_t i = 0; i < neuron_count; i++)
		{
			INeuron* current_neuron = neurons[i];
			current_neuron->ExecuteStore(network_activations, execution_results, t_index);
		}
	}

public:
	double* Execute(double* X, size_t t_count = 1, bool delete_memory = true)
	{
		double* output = new double[output_length * t_count];
		double* network_activations = new double[t_count * (input_length + neuron_count)];
		for (size_t t = 0; t < t_count; t++)
		{
			size_t per_t_modifier = (neuron_count + input_length) * t;
			for (size_t i = 0; i < input_length; i++)
			{
				network_activations[i + per_t_modifier] = X[i + t * input_length];
			}

			for (size_t i = 0; i < neuron_count; i++)
			{
				INeuron* current_neuron = neurons[i];
				current_neuron->Execute(network_activations, t);
			}

			for (size_t i = 0; i < output_length; i++)
			{
				output[t * output_length + i] =
					network_activations[per_t_modifier + neuron_count + input_length - output_length + i];
			}
		}

		if (delete_memory)
			for (size_t i = 0; i < neuron_count; i++)
			{
				neurons[i]->DeleteMemory();
			}

		delete[] network_activations;
		return output;
	}

	/// <summary>
	/// Works as a batch for non-recurrent neurons and for recurrent neurons it works as training over t
	/// </summary>
	void Supervised_batch(double* X, double* Y, double learning_rate, size_t t_count, Cost::CostFunction cost_function, bool delete_memory = true)
	{
		double* costs = new double[t_count * (neuron_count + input_length)];
		double* gradients = new double[t_count * gradients_value_count];
		double* activations = new double[t_count * (input_length + neuron_count)];
		double* execution_results = new double[t_count * execution_results_value_count];

		// Inference
		for (size_t t = 0; t < t_count; t++)
		{
			ExecuteStore(X, activations, execution_results, t);

			size_t per_t_Y_addition = output_length * t;

			size_t per_t_addition = t * (neuron_count + input_length);
			size_t current_output_start = per_t_addition + neuron_count + input_length - output_length;
			for (size_t i = 0; i < output_length; i++)
			{
				size_t current_output_index = current_output_start + i;
				costs[current_output_index] = Derivatives::DerivativeOf(activations[current_output_index], Y[per_t_Y_addition + i], cost_function);
			}
		}

		// Gradient calculation
		for (int i = neuron_count - 1; i >= 0; i--)
		{
			neurons[i]->GetGradients(gradients, costs, execution_results, activations, t_count);
		}

		for (size_t i = 0; i < neuron_count; i++)
		{
			neurons[i]->SubtractGradients(gradients, learning_rate, t_count);
		}

		if (delete_memory)
			for (size_t i = 0; i < neuron_count; i++)
			{
				neurons[i]->DeleteMemory();
			}

		delete[] costs;
		delete[] gradients;
		delete[] activations;
		delete[] execution_results;
	}

	void free()
	{
		for (size_t i = 0; i < neuron_count; i++)
		{
			neurons[i]->Free();
		}
		delete[] neurons;
	}
};
