#include <stdio.h>
#include <thread>
#include <vector>
#include <functional>
#include <chrono>

#include "INeuron.h"
#include "Derivatives.h"
#include "Cost.h"

// Neurons
#include "DenseNeuron.h"
#include "DenseLSTM.h"

#pragma once
/// <summary>
/// Note: For proper NEAT algorithm only use NEAT-based neurons, layers after NEAT are symbolical
/// </summary>
class NN
{
private:
	INeuron** neurons = 0;
	/// <summary>
	/// starting_shape if it contains NeatNeurons
	/// </summary>
	size_t* shape = 0;
	size_t shape_length = -1;

	/// <summary>
	/// Used as a cap for the maximum number of neurons for evolution algorithms
	/// </summary>
	size_t max_neuron_count = -1;
	
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

	enum LearningRateOptimizators
	{
		None,
		LearningEffectiveness,
		InverseLearningEffectiveness
	};

private:
	int* neuron_types = 0;

public:
	/// <summary>
	/// Remember to save and load recurrent states, learning and executing will modify recurrent state
	/// </summary>
	/// <param name="input_layer_length">This layer is not instantiated as neurons</param>
	/// <param name="neuron_types">By leaving the parameter as null you must save neuron types externally in order to save the network, else you don't have to provide it</param>
	/// <param name="max_neuron_count">Used as a cap of neurons for neuroevolution</param>
	NN(INeuron** neurons, size_t neuron_count, size_t input_layer_length, size_t output_layer_length, size_t* network_shape, size_t shape_length,
		NeuronTypeIdentifier* neuron_types = 0, bool free_neuron_types = true, int* parsed_neuron_types = 0, size_t max_neuron_count = 0, bool populate_values = true)
	{
		max_neuron_count += neuron_count - max_neuron_count * (max_neuron_count < neuron_count);
		input_length = input_layer_length;
		output_length = output_layer_length;
		this->neuron_count = neuron_count;
		this->max_neuron_count = max_neuron_count;
		this->shape = network_shape;
		this->shape_length = shape_length;

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

	double AdjustLearningRate(double original_learning_rate, LearningRateOptimizators optimize_based_of, double* previous_cost, double current_cost)
	{
		// TODO: solve bugs
		double cost_difference;
		switch (optimize_based_of)
		{
		case NN::None:
			return original_learning_rate;
			break;
		case NN::LearningEffectiveness:
			if (previous_cost == 0)
				return original_learning_rate;
			
			if (!current_cost)
				return original_learning_rate;

			/*
				Scenario:
					Previous_cost > Current_cost
					(Learning occured)
					Means:
						Learning rate should go up based on the difference
						or else:
						Learning_rate should go down based on the difference
			*/

			cost_difference = *previous_cost - current_cost;
			current_cost -= (current_cost * 2) * (cost_difference < 0);
			return original_learning_rate + (*previous_cost / current_cost);
		case NN::InverseLearningEffectiveness:
			if (previous_cost == 0)
				return original_learning_rate;


			if (*previous_cost == 0)
				return original_learning_rate;

			cost_difference = current_cost - *previous_cost;
			current_cost -= (current_cost * 2) * (cost_difference < 0);
			return original_learning_rate + (current_cost / *previous_cost);

		default:
			throw std::exception("Learning rate optimizator not implemented");
		}
	}

private:
	void ExecuteStore(double* X, double* network_activations, double* execution_results, bool use_multiprocessing, size_t t_index = 0)
	{
		for (size_t i = 0; i < input_length; i++)
		{
			network_activations[i + (input_length + neuron_count) * t_index] = X[i + t_index * input_length];

		}

		if (use_multiprocessing)
		{
			std::vector<std::thread> threads = std::vector<std::thread>();
			size_t neuron_i = 0;
			for (size_t i = 1; i < shape_length; i++)
			{
				size_t layer_length = shape[i];
				for (size_t j = 0; j < layer_length; j++, neuron_i++)
				{
					threads.push_back(std::thread(&INeuron::ExecuteStore, neurons[neuron_i], network_activations, execution_results, t_index));
				}
				for (size_t j = 0; j < layer_length; j++)
				{
					threads[j].join();
				}
				threads.clear();
			}
		}
		else
			for (size_t i = 0; i < neuron_count; i++)
			{
				INeuron* current_neuron = neurons[i];
				current_neuron->ExecuteStore(network_activations, execution_results, t_index);
			}
	}

public:
	double* Execute(double* X, size_t t_count = 1, bool delete_memory = true, bool use_multithreading = true)
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

			if (use_multithreading)
			{
				auto threads = std::vector<std::thread>();
				size_t neuron_i = 0;
				for (size_t i = 1; i < shape_length; i++)
				{
					size_t layer_length = shape[i];
					for (size_t j = 0; j < layer_length; j++, neuron_i++)
					{
						threads.push_back(std::thread(&INeuron::Execute, neurons[neuron_i], network_activations, t));
					}
					for (size_t j = 0; j < layer_length; j++)
					{
						threads[j].join();
					}
					threads.clear();
				}
			}
			else
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
	double Supervised_batch(double* X, double* Y, double* learning_rate, bool modify_learning_rate, size_t t_count, Cost::CostFunction cost_function, LearningRateOptimizators optimizator, double* previous_cost, bool use_multithreading = false, size_t X_start_i = 0, size_t Y_start_i = 0, bool delete_memory = true, double dropout_rate = 0)
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

		std::vector<std::thread> threads = std::vector<std::thread>();

		// Inference
		double cost = 0;
		for (size_t t = 0; t < t_count; t++)
		{
			TrainingInference(current_X, current_Y, activations, execution_results, costs, t, &cost, cost_function, use_multithreading);
		}
		cost /= t_count;

		delete[] current_X;
		delete[] current_Y;

		// Gradient calculation
		size_t neuron_i = neuron_count - 1;
		for (size_t layer_i = shape_length - 1; layer_i >= 1; layer_i--)
		{
			size_t layer_trained_neurons = 0;
			for (size_t i = 0; i < shape[i]; i++, neuron_i--)
			{
				if (ValueGeneration::NextDouble() < dropout_rate && ((neuron_count - 1 - i) > output_length))
					continue;

				if (use_multithreading)
				{
					threads.push_back(std::thread(&INeuron::GetGradients, neurons[neuron_i], gradients, costs, execution_results, activations, t_count));
					layer_trained_neurons++;
				}
				else
				{
					neurons[neuron_i]->GetGradients(gradients, costs, execution_results, activations, t_count);
				}
			}
			for (size_t i = 0; i < layer_trained_neurons; i++)
			{
				threads[i].join();
			}
			threads.clear();
		}

		double optimized_learning_rate = AdjustLearningRate(*learning_rate, optimizator, previous_cost, cost);
		*learning_rate += (optimized_learning_rate - *learning_rate) * modify_learning_rate;
		*learning_rate += (-*learning_rate + .001) * (*learning_rate <= 0);

		for (size_t i = 0; i < neuron_count; i++)
		{
			neurons[i]->SubtractGradients(gradients, optimized_learning_rate, t_count);
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

		return cost;
	}

	void TrainingInference(double* current_X, double* current_Y, double* activations, double* execution_results, double* costs, size_t t, double* cost, Cost::CostFunction cost_function, bool use_multiprocessing)
	{
		double current_t_cost = 0;

		ExecuteStore(current_X, activations, execution_results, use_multiprocessing, t);

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
		*cost += current_t_cost;
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
		size_t metadata[7]
		{
			neuron_count,
			input_length,
			output_length,
			execution_results_value_count,
			gradients_value_count,
			shape_length,
			max_neuron_count
		};

		FILE* neuron_type_file;
		if (fopen_s(&neuron_type_file, (path_with_no_extension + GetNeuronTypeFileExtension()).data(), "wb"))
			throw std::exception("File cannot be opened");
		fwrite(&metadata, sizeof(size_t), 7, neuron_type_file);
		fwrite(neuronTypes, sizeof(int), neuron_count, neuron_type_file);
		fwrite(shape, sizeof(size_t), shape_length, neuron_type_file);
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
		size_t metadata[7]{};
		//	neuron_count,
		//	input_length,
		//	output_length,
		//	execution_results_value_count,
		//	gradients_value_count
		//	shape_length
		//  max_neuron_count

		FILE* nt_file;
		if (fopen_s(&nt_file, (path_with_no_extension + GetNeuronTypeFileExtension()).data(), "rb"))
			throw std::exception("File cannot be opened");

		fread(&metadata, sizeof(size_t), 7, nt_file);

		size_t neuron_count = metadata[0];
		size_t input_length = metadata[1];
		size_t output_length = metadata[2];
		size_t execution_results_value_count = metadata[3];
		size_t gradients_value_count = metadata[4];
		size_t shape_length = metadata[5];
		size_t max_neuron_count = metadata[6];

		int* neuron_types = new int[neuron_count];
		fread(neuron_types, sizeof(int), neuron_count, nt_file);

		size_t* shape = new size_t[shape_length];
		fread(shape, sizeof(size_t), shape_length, nt_file);

		fclose(nt_file);

		FILE* nn_file; 
		if (fopen_s(&nn_file, (path_with_no_extension + GetNNFileExtension()).data(), "rb"))
			throw std::exception("File cannot be opened");

		INeuron** neurons = new INeuron * [max_neuron_count];

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

		NN* out = new NN(neurons, neuron_count, input_length, output_length, shape, shape_length, 0, false, neuron_types, max_neuron_count, false);
		out->gradients_value_count = gradients_value_count;
		out->execution_results_value_count = execution_results_value_count;

		return out;
	}

	INeuron* GetNeuron(size_t neuron_i)
	{
		return neurons[neuron_i];
	}

	void free(bool free_shape = true)
	{
		for (size_t i = 0; i < neuron_count; i++)
		{
			neurons[i]->Free();
		}
		delete[] neurons;
		delete[] neuron_types;
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

public:
	static void SetValueGenerationSeed()
	{
		srand(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
	}
};
