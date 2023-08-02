#include <list>
#include <tuple>

#include "INeuron.h"
#include "Cost.h"

#pragma once
class NN
{
private:
	INeuron** neurons = 0;
	size_t neuron_count = -1;
	size_t execution_results_value_count = -1;
	size_t input_length = -1;
	size_t output_length = -1;

public:
	/// <param name="input_layer_length">This layer is not instantiated as neurons</param>
	NN(INeuron** neurons, size_t neuron_count, size_t input_layer_length, size_t output_layer_length)
	{
		size_t network_execution_results_value_count = 0;
		for (size_t i = 0; i < neuron_count; i++)
		{
			INeuron* current_neuron = neurons[i];

			// Set
			current_neuron->self_execution_results_start_i = network_execution_results_value_count;
			current_neuron->connections->self_gradients_start_i = network_execution_results_value_count;
			current_neuron->connections->network_neuron_count = neuron_count;


			//Get
			network_execution_results_value_count += current_neuron->GetNeuronWrittenGradientCount();
			network_execution_results_value_count += current_neuron->connections->GetWeightCount();
		}

		for (size_t i = 0; i < neuron_count; i++)
		{
			neurons[i]->connections->network_execution_results_value_count = network_execution_results_value_count;
		}

		this->neuron_count = neuron_count;
		this->execution_results_value_count = network_execution_results_value_count;
		input_layer_length = input_layer_length;
		output_length = output_layer_length;
	}

private:
	void ExecuteStore(double* X, double* network_activations, double* execution_results, size_t t_index = 0)
	{
		for (size_t i = 0; i < input_length; i++)
		{
			network_activations[i + neuron_count * t_index] = X[i + t_index * input_length];
		}
		for (size_t i = 0; i < neuron_count; i++)
		{
			INeuron* current_neuron = neurons[i];
			current_neuron->ExecuteStore(network_activations, execution_results, t_index);
		}
	}

public:
	double* Execute(double* X, size_t t_count = 1)
	{
		double* network_activations = new double[t_count * (input_length + neuron_count)];
		double* execution_results = new double[t_count * (execution_results_value_count)];
		for (size_t t = 0; t < t_count; t++)
		{
			for (size_t i = 0; i < input_length; i++)
			{
				network_activations[i + neuron_count * t] = X[i + t * input_length];
			}
			for (size_t i = 0; i < neuron_count; i++)
			{
				INeuron* current_neuron = neurons[i];
				current_neuron->ExecuteStore(network_activations, execution_results, t);
			}
		}
		delete[] network_activations;
		delete[] execution_results;
	}

	/// <summary>
	/// Also recommended for recurrent layers_not_including_input_layer, if there are none recurrent layers_not_including_input_layer this will function as a batch
	/// </summary>
	void Supervised_Train(size_t t_count, double** X, double** Y, Cost::CostFunction cost_function, double learning_rate, size_t starting_i = 0)
	{
	}

	/// <summary>
	/// Perfect for making batches without recurrent layers_not_including_input_layer
	/// </summary>
	void Supervised_Train(double** X, double** Y, size_t step_count, size_t batch_size, double learning_rate)
	{
	}

	/// <summary>
	/// Perfect for making batches with recurrent layers_not_including_input_layer
	/// </summary>
	void Supervised_Train(double*** X, double*** Y, size_t step_count, size_t* t_count_per_step, double learning_rate, size_t batch_size = 1)
	{
	}

	void free()
	{
	}
};
