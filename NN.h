#include <list>
#include <tuple>

#include "INeuron.h"
#include "Cost.h"

#pragma once
class NN
{
private:
	INeuron** neurons = 0;

public:
	NN(INeuron** neurons, size_t neuron_count)
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
	}

private:
	std::tuple<double*, double*> ExecuteStore(double* X)
	{
	}

public:
	double* Execute(double* X)
	{
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
