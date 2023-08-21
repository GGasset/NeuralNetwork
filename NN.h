#include <stdio.h>

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
	enum NeuronTypeIdentifier
	{
		DenseNeuronId = 0,
		DenseLSTMId = 1
	};

private:
	int* neuron_types = 0;

public:
	/// <param name="input_layer_length">This layer is not instantiated as neurons</param>
	/// <param name="neuron_types">By leaving the parameter as null you must save neuron types externally in order to save the network, else you don't have to provide it</param>
	NN(INeuron** neurons, size_t neuron_count, size_t input_layer_length, size_t output_layer_length, NeuronTypeIdentifier* neuron_types = 0, bool populate_values = true)
	{
		input_length = input_layer_length;
		output_length = output_layer_length;
		this->neuron_count = neuron_count;

		this->neurons = neurons;
		if (populate_values)
		{
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

		if (neuron_types)
		{
			this->neuron_types = new int[neuron_count];
			for (size_t i = 0; i < neuron_count; i++)
			{
				this->neuron_types[i] = neuron_types[i];
			}
			delete[] neuron_types;
		}
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
	void Supervised_batch(double* X, double* Y, double learning_rate, size_t t_count, Cost::CostFunction cost_function, size_t X_start_i = 0, size_t Y_start_i = 0, bool delete_memory = true, double dropout_rate = 0)
	{
		size_t current_X_size = input_length * t_count;
		double* current_X = new double[current_X_size];
		for (size_t i = 0; i < current_X_size; i++)
		{
			current_X[i] = X[i + X_start_i];
		}

		size_t current_Y_size = output_length * t_count;
		double* current_Y = new double[current_Y_size];
		for (size_t i = 0; i < current_Y_size; i++)
		{
			current_Y[i] = Y[i + Y_start_i];
		}

		size_t single_value_for_neurons_count = t_count * (neuron_count + input_length);
		double* costs = new double[single_value_for_neurons_count];
		double* gradients = new double[t_count * gradients_value_count];
		double* activations = new double[single_value_for_neurons_count];
		double* execution_results = new double[t_count * execution_results_value_count];

		for (size_t i = 0; i < (t_count * gradients_value_count); i++)
		{
			gradients[i] = 0;
		}

		for (size_t i = 0; i < single_value_for_neurons_count; i++)
		{
			costs[i] = activations[i] = 0;
		}

		// There is a false positive as the warning says that per_t_Y_addition may be null, it must have 0 as a value to properly function
#pragma warning(push)
#pragma warning(disable:6385)
		// Inference
		for (size_t t = 0; t < t_count; t++)
		{
			ExecuteStore(current_X, activations, execution_results, t);

			size_t per_t_Y_addition = output_length * t;

			size_t per_t_addition = t * (neuron_count + input_length);
			size_t current_output_start = per_t_addition + neuron_count + input_length - output_length;
			for (size_t i = 0; i < output_length; i++)
			{
				size_t current_output_index = current_output_start + i;
				costs[current_output_index] = Derivatives::DerivativeOf(activations[current_output_index], current_Y[per_t_Y_addition + i], cost_function);
			}
		}
#pragma warning(pop)

		delete[] current_X;
		delete[] current_Y;

		// Gradient calculation
		for (int i = neuron_count - 1; i >= 0; i--)
		{
			if (ValueGeneration::NextDouble() < dropout_rate && ((neuron_count - 1 - i) > output_length))
				continue;

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

	void Save(std::string path_with_no_extension)
	{
		if (neuron_types == 0)
			throw std::exception("No info about neuron_types provided, please use another overload of this method.\nYou may also save that info at the constructor");

		Save(path_with_no_extension, neuron_types);
	}

	void Save(std::string path_with_no_extension, int* neuronTypes)
	{
		size_t metadata[5]
		{
			neuron_count,
			input_length,
			output_length,
			execution_results_value_count,
			gradients_value_count
		};

		FILE* neuron_type_file = fopen((path_with_no_extension + GetNeuronTypeFileExtension()).data(), "w");
		fwrite(&metadata, sizeof(size_t), 5, neuron_type_file);
		fwrite(neuronTypes, sizeof(int), neuron_count, neuron_type_file);
		fclose(neuron_type_file);

		FILE* nn_file = fopen((path_with_no_extension + GetNNFileExtension()).data(), "w");
		for (size_t i = 0; i < neuron_count; i++)
		{
			INeuron* current_neuron = neurons[i];

			size_t neuron_size;
			size_t connections_size;
			size_t weight_count = current_neuron->connections->GetWeightCount();
			switch (neuronTypes[i])
			{
			case 0:
				neuron_size = sizeof(DenseNeuron);
				connections_size = sizeof(DenseConnections);
				break;
			case 1:
				neuron_size = sizeof(DenseLSTM);
				connections_size = sizeof(DenseConnections);
				break;
			default:
				throw std::exception("NeuronType not implemented");
			}
			fwrite(current_neuron, neuron_size, 1, nn_file);
			fwrite(current_neuron->connections, connections_size, 1, nn_file);
			fwrite(current_neuron->connections->GetWeights(), sizeof(double), weight_count, nn_file);
		}
		fclose(nn_file);
	}

	static NN Load(std::string path_with_no_extension)
	{
	}

private:
	static std::string GetNeuronTypeFileExtension()
	{
		return std::string(".nt");
	}

	static std::string GetNNFileExtension()
	{
		return std::string(".nn");
	}
};
