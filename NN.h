#include <stdio.h>
#include <iostream>

#include "INeuron.h"
#include "Derivatives.h"
#include "Cost.h"

// Neurons
#include "DenseNeuron.h"
#include "DenseLSTM.h"

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
	NN(INeuron** neurons, size_t neuron_count, size_t input_layer_length, size_t output_layer_length, NeuronTypeIdentifier* neuron_types = 0, bool free_neuron_types = true, int* parsed_neuron_types = 0, bool populate_values = true)
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
			if (free_neuron_types)
				delete[] neuron_types;
		}
		else if (parsed_neuron_types)
		{
			this->neuron_types = parsed_neuron_types;
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
	/// Works as a batch for non-recurrent neurons and for recurrent neurons it works as training over t. Returns: Mean output cost averaged over t of the average neuron_cost
	/// </summary>
	double Supervised_batch(double* X, double* Y, double learning_rate, size_t t_count, Cost::CostFunction cost_function, size_t X_start_i = 0, size_t Y_start_i = 0, bool delete_memory = true, double dropout_rate = 0)
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
		double cost = 0;
		for (size_t t = 0; t < t_count; t++)
		{
			double current_t_cost = 0;

			ExecuteStore(current_X, activations, execution_results, t);

			size_t per_t_Y_addition = output_length * t;

			size_t per_t_addition = t * (neuron_count + input_length);
			size_t current_output_start = per_t_addition + neuron_count + input_length - output_length;
			for (size_t i = 0; i < output_length; i++)
			{
				size_t current_output_index = current_output_start + i;
				costs[current_output_index] = Derivatives::DerivativeOf(activations[current_output_index], current_Y[per_t_Y_addition + i], cost_function);
				current_t_cost += Cost::GetCostOf(activations[current_output_index], current_Y[per_t_Y_addition + i], cost_function);
			}
			current_t_cost /= output_length;
			cost += current_t_cost;
		}
		cost /= t_count;

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

		return cost
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

	void Save(std::string path_with_no_extensions, NeuronTypeIdentifier *neuron_types)
	{
		if (neuron_types == 0)
			throw std::exception("neuron_types is null");

		int* parsed_neuron_types = new int[neuron_count];
		for (size_t i = 0; i < neuron_count; i++)
		{
			parsed_neuron_types[i] = neuron_types[i];
		}

		Save(path_with_no_extensions, parsed_neuron_types);
		delete[] parsed_neuron_types;
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

		FILE* neuron_type_file;
		if (fopen_s(&neuron_type_file, (path_with_no_extension + GetNeuronTypeFileExtension()).data(), "wb"))
			throw std::exception("File cannot be opened");
		fwrite(&metadata, sizeof(size_t), 5, neuron_type_file);
		fwrite(neuronTypes, sizeof(int), neuron_count, neuron_type_file);
		fclose(neuron_type_file);

		FILE* nn_file;
		if (fopen_s(&nn_file, (path_with_no_extension + GetNNFileExtension()).data(), "wb"))
			throw std::exception("File cannot be opened");

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

	static NN* Load(std::string path_with_no_extension)
	{
		size_t metadata[5]{};
		//	neuron_count,
		//	input_length,
		//	output_length,
		//	execution_results_value_count,
		//	gradients_value_count

		FILE* nt_file;
		if (fopen_s(&nt_file, (path_with_no_extension + GetNeuronTypeFileExtension()).data(), "rb"))
			throw std::exception("File cannot be opened");

		fread(&metadata, sizeof(size_t), 5, nt_file);

		size_t neuron_count = metadata[0];
		size_t input_length = metadata[1];
		size_t output_length = metadata[2];
		size_t execution_results_value_count = metadata[3];
		size_t gradients_value_count = metadata[4];

		int* neuron_types = new int[neuron_count];
		fread(neuron_types, sizeof(int), neuron_count, nt_file);
		fclose(nt_file);

		FILE* nn_file; 
		if (fopen_s(&nn_file, (path_with_no_extension + GetNNFileExtension()).data(), "rb"))
			throw std::exception("File cannot be opened");

		INeuron** neurons = new INeuron * [neuron_count];

		INeuron* neuron;
		double* weights;
		size_t weight_count;
		for (size_t i = 0; i < neuron_count; i++)
		{
			switch ((NeuronTypeIdentifier)neuron_types[i])
			{
			case DenseNeuronId:
				neuron = (DenseNeuron*)malloc(sizeof(DenseNeuron));
				fread(neuron, sizeof(DenseNeuron), 1, nn_file);
				
				neuron->connections = (DenseConnections*)malloc(sizeof(DenseConnections));
				fread(neuron->connections, sizeof(DenseConnections), 1, nn_file);

				neurons[i] = neuron;
				break;

			case DenseLSTMId:
				neuron = (DenseLSTM*)malloc(sizeof(DenseLSTM));
				fread(neuron, sizeof(DenseLSTM), 1, nn_file);

				neuron->connections = (DenseConnections*)malloc(sizeof(DenseConnections));
				fread(neuron->connections, sizeof(DenseConnections), 1, nn_file);

				neurons[i] = neuron;
				break;
			default:
				throw std::exception("Neuron not implemented for loading");
			}

			weight_count = neuron->connections->GetWeightCount();
			weights = new double[weight_count];
			fread(weights, sizeof(double), weight_count, nn_file);
			neuron->connections->SetWeights(weights);
			weights = 0;
		}
		fclose(nn_file);

		NN* out = new NN(neurons, neuron_count, input_length, output_length, 0, neuron_types, false);
		out->gradients_value_count = gradients_value_count;
		out->execution_results_value_count = execution_results_value_count;

		return out;
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
