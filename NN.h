#include <stdio.h>
#include <thread>
#include <vector>
#include <functional>
#include <chrono>
#include <string>

#include "INeuron.h"
#include "Derivatives.h"
#include "Cost.h"
#include "EvolutionMetaData.h"

// Neurons
#include "DenseNeuron.h"
#include "DenseLSTM.h"
#include "NEATNeuron.h"
#include "NEATLSTM.h"

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
	size_t max_layer_count = -1;

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
	static const int NeuronTypeID_per_connection_type_max_value_count = 1000;

	/// <summary>
	/// must be an integer | DenseConnections based Neurons: 0-999 | NEATConnections based Neurons: 1000-1999
	/// </summary>
	enum NeuronTypeIdentifier
	{
		DenseNeuronId = 0,
		DenseLSTMId = 1,
		NEATNeuronId = 1000,
		NEATLSTMId = 1001
	};

	enum LearningRateOptimizators
	{
		None,
		LearningEffectiveness,
		InverseLearningEffectiveness,
		HighCostHighLearning
	};

private:
	int* neuron_types = 0;
	EvolutionMetaData* evolution_metadata = 0;

public:
	/// <summary>
	/// Remember to save and load recurrent states, learning and executing will modify recurrent state
	/// </summary>
	/// <param name="input_layer_length">This layer is not instantiated as neurons</param>
	/// <param name="neuron_types">By leaving the parameter as null you must save neuron types externally in order to save the network, else you don't have to provide it</param>
	/// <param name="max_neuron_count">Used as a cap of neurons for neuroevolution</param>
	NN(INeuron** neurons, size_t neuron_count, size_t input_layer_length, size_t output_layer_length, size_t* network_shape, size_t layer_count,
		bool free_network_shape = false, NeuronTypeIdentifier* neuron_types = 0, int* parsed_neuron_types = 0, bool free_neuron_types = true, size_t max_neuron_count = 0, size_t max_layer_count = 0, EvolutionMetaData* evolution_values = 0, bool populate_values = true)
	{
		max_neuron_count += (neuron_count - max_neuron_count) * (max_neuron_count < neuron_count);
		max_layer_count += (layer_count - max_layer_count) * (max_layer_count < layer_count);

		input_length = input_layer_length;
		output_length = output_layer_length;
		this->neuron_count = neuron_count;
		this->max_neuron_count = max_neuron_count;
		this->shape = network_shape;
		this->shape_length = layer_count;
		this->max_layer_count = max_layer_count;
		this->evolution_metadata = new EvolutionMetaData(*evolution_values);

		this->neurons = neurons;
		if (populate_values)
		{
			PopulateAutomaticallySetValues();
		}

		if (neuron_types)
		{
			this->neuron_types = new int[max_neuron_count];
			for (size_t i = 0; i < neuron_count; i++)
			{
				this->neuron_types[i] = neuron_types[i];
			}
			if (free_neuron_types)
				delete[] neuron_types;
		}
		else if (parsed_neuron_types)
		{
			this->neuron_types = new int[max_neuron_count];
			for (size_t i = 0; i < neuron_count; i++)
			{
				this->neuron_types[i] = parsed_neuron_types[i];
			}
			if (free_neuron_types)
				delete[] parsed_neuron_types;
		}
	}

	void PopulateAutomaticallySetValues()
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

	double AdjustLearningRate(double original_learning_rate, LearningRateOptimizators optimize_based_of, double* previous_cost, double current_cost)
	{
		double cost_difference;
		double optimization;
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

			optimization = .5 - (current_cost - *previous_cost);
			optimization += (-optimization) * (optimization < -original_learning_rate);

			return original_learning_rate + optimization;
			//return original_learning_rate + (current_cost / *previous_cost);
		case NN::HighCostHighLearning:
			return original_learning_rate + current_cost;
			break;
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
	double SupervisedBatch(double* X, double* Y, size_t t_count, Cost::CostFunction cost_function, double* learning_rate, LearningRateOptimizators optimizator, bool modify_learning_rate = false, double* previous_cost = 0, bool use_multithreading = true, double dropout_rate = 0, bool delete_memory = true, size_t X_start_i = 0, size_t Y_start_i = 0)
	{
		double* costs, *gradients, *activations, *execution_results;
		costs = gradients = activations = execution_results = 0;
		SetupTrainingVariables(&costs, &gradients, &activations, &execution_results, t_count);

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

		double cost = 0;
		for (size_t t = 0; t < t_count; t++)
			TrainingInference(current_X, current_Y, activations, execution_results, costs, t, &cost, cost_function, use_multithreading);

		delete[] current_X;
		delete[] current_Y;

		CalculateGradients(gradients, costs, execution_results, activations, t_count, use_multithreading, delete_memory, dropout_rate);

		SubtractGradients(gradients, t_count, learning_rate, previous_cost, cost, optimizator, modify_learning_rate, use_multithreading);

		return cost;
	}

	void SubtractGradients(double* gradients, size_t t_count, double* learning_rate, double* previous_cost, double mean_cost, LearningRateOptimizators optimizator, bool modify_learning_rate = false, bool use_multithreading = true)
	{
		double optimized_learning_rate = AdjustLearningRate(*learning_rate, optimizator, previous_cost, mean_cost);
		*learning_rate += (optimized_learning_rate - *learning_rate) * modify_learning_rate;
		*learning_rate += (-*learning_rate + .001) * (*learning_rate <= 0);

		if (use_multithreading)
		{
			std::vector<std::thread> threads = std::vector<std::thread>();
			for (size_t i = 0; i < neuron_count; i++)
				threads.push_back(std::thread(&INeuron::SubtractGradients, neurons[i], gradients, optimized_learning_rate, t_count));

			for (size_t i = 0; i < neuron_count; i++)
				threads[i].join();
		}
		else
			for (size_t i = 0; i < neuron_count; i++)
			{
				neurons[i]->SubtractGradients(gradients, optimized_learning_rate, t_count);
			}

		delete[] gradients;
	}

	void CalculateGradients(double* gradients, double* costs, double* execution_results, double* activations, size_t t_count,
		bool use_multithreading = false, bool delete_memory = true, double dropout_rate = 0)
	{
		std::vector<std::thread> threads = std::vector<std::thread>();
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


		if (delete_memory)
			for (size_t i = 0; i < neuron_count; i++)
			{
				neurons[i]->DeleteMemory();
			}

		delete[] costs;
		delete[] execution_results;
		delete[] activations;
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

	void SetupTrainingVariables(double** costs, double** gradients, double** activations, double** execution_results, size_t t_count)
	{
		size_t total_neurons_count = t_count * (neuron_count + input_length);
		size_t total_gradient_value_count = t_count * gradients_value_count;

		*costs = new double[total_neurons_count];
		*gradients = new double[total_gradient_value_count];
		*activations = new double[total_neurons_count];
		*execution_results = new double[t_count * execution_results_value_count];

		for (size_t i = 0; i < total_gradient_value_count; i++)
		{
			(*gradients)[i] = 0;
		}

		for (size_t i = 0; i < total_neurons_count; i++)
		{
			(*costs)[i] = (*activations)[i] = 0;
		}

	}

	/// <summary>
	/// Use only with all neurons having derived connections from NEATConnections or connections with the proper methods implemented | 
	/// evolution_metadata must not be equal NULL to avoid weird unpredictable behaivor
	/// </summary>
	void Evolve()
	{
		double max_variation = evolution_metadata->max_weight_mutation * 2;
		double bias = evolution_metadata->max_weight_mutation;
		for (size_t i = 0; i < neuron_count; i++)
		{
			INeuron* current_neuron = neurons[i];
			IConnections* current_connection = current_neuron->connections;
			
			current_neuron->SetBias(current_neuron->GetBias() + (ValueGeneration::NextDouble() * max_variation - bias) * (ValueGeneration::NextDouble() < evolution_metadata->weight_mutation_probability));
			
			size_t weight_count = current_connection->GetWeightCount();
			double* weights = current_connection->GetWeights();
			for (size_t i = 0; i < weight_count; i++)
				weights[i] += 
					(ValueGeneration::NextDouble() * max_variation - bias) 
					* (ValueGeneration::NextDouble() < evolution_metadata->weight_mutation_probability);
		}

		if (evolution_metadata->new_neuron_chance > ValueGeneration::NextDouble())
			AugmentTopology();

		EvolveMetadata();
	}

	void EvolveMetadata()
	{
		evolution_metadata->max_weight_mutation +=
			(evolution_metadata->max_weight_mutation_max_mutation * 2 * ValueGeneration::NextDouble() - evolution_metadata->max_weight_mutation_max_mutation)
			* (evolution_metadata->max_weight_mutation_mutation_chance > ValueGeneration::NextDouble());
		
		evolution_metadata->weight_mutation_probability +=
			(evolution_metadata->weight_mutation_probability_max_mutation * 2 * ValueGeneration::NextDouble() - evolution_metadata->weight_mutation_probability_max_mutation)
			* (evolution_metadata->weight_mutation_probability_mutation_chance > ValueGeneration::NextDouble());

		evolution_metadata->new_neuron_chance +=
			(evolution_metadata->new_neuron_chance_max_mutation * 2 * ValueGeneration::NextDouble() - evolution_metadata->new_neuron_chance_max_mutation)
			* (evolution_metadata->new_neuron_chance_mutation_probability > ValueGeneration::NextDouble());

		evolution_metadata->new_neuron_in_new_layer_chance +=
			(evolution_metadata->new_neuron_in_new_layer_chance_max_mutation * 2 * ValueGeneration::NextDouble() - evolution_metadata->new_neuron_in_new_layer_chance_max_mutation)
			* (evolution_metadata->new_neuron_in_new_layer_chance_mutation_probability > ValueGeneration::NextDouble());



		evolution_metadata->max_weight_mutation_max_mutation +=
			(evolution_metadata->max_mutation_of_max_weight_mutation_max_mutation * 2 * ValueGeneration::NextDouble() - evolution_metadata->max_mutation_of_max_weight_mutation_max_mutation)
			* (evolution_metadata->max_weight_mutation_mutation_chance > ValueGeneration::NextDouble());

		evolution_metadata->weight_mutation_probability_max_mutation +=
			(evolution_metadata->max_mutation_of_weight_mutation_probability_max_mutation * 2 * ValueGeneration::NextDouble() - evolution_metadata->max_mutation_of_weight_mutation_probability_max_mutation)
			* (evolution_metadata->mutation_chance_of_weight_mutation_probability_max_mutation > ValueGeneration::NextDouble());

		evolution_metadata->new_neuron_chance_max_mutation +=
			(evolution_metadata->max_mutation_of_new_neuron_chance_max_mutation * 2 * ValueGeneration::NextDouble() - evolution_metadata->max_mutation_of_new_neuron_chance_max_mutation)
			* (evolution_metadata->mutation_chance_of_new_neuron_chance_max_mutation > ValueGeneration::NextDouble());

		evolution_metadata->new_neuron_in_new_layer_chance_max_mutation +=
			(evolution_metadata->max_mutation_of_new_neuron_in_new_layer_chance_max_mutation * 2 * ValueGeneration::NextDouble() - evolution_metadata->max_mutation_of_new_neuron_in_new_layer_chance_max_mutation)
			* (evolution_metadata->mutation_chance_of_new_neuron_in_new_layer_chance_max_mutation > ValueGeneration::NextDouble());



		evolution_metadata->max_weight_mutation_mutation_chance +=
			(evolution_metadata->max_mutation_of_max_weight_mutation_mutation_chance * 2 * ValueGeneration::NextDouble() - evolution_metadata->max_mutation_of_max_weight_mutation_mutation_chance)
			* (evolution_metadata->mutation_chance_of_max_weight_mutation_mutation_chance > ValueGeneration::NextDouble());

		evolution_metadata->weight_mutation_probability_mutation_chance +=
			(evolution_metadata->max_mutation_of_weight_mutation_probability_mutation_chance * 2 * ValueGeneration::NextDouble() - evolution_metadata->max_mutation_of_weight_mutation_probability_mutation_chance)
			* (evolution_metadata->mutation_chance_of_weight_mutation_probability_mutation_chance > ValueGeneration::NextDouble());

		evolution_metadata->new_neuron_chance_mutation_probability +=
			(evolution_metadata->max_mutation_of_new_neuron_chance_mutation_probability * 2 * ValueGeneration::NextDouble() - evolution_metadata->max_mutation_of_new_neuron_chance_mutation_probability)
			* (evolution_metadata->mutation_chance_of_new_neuron_chance_mutation_probability > ValueGeneration::NextDouble());

		evolution_metadata->new_neuron_in_new_layer_chance_mutation_probability +=
			(evolution_metadata->max_mutation_of_new_neuron_in_new_layer_chance_mutation_probability * 2 * ValueGeneration::NextDouble() - evolution_metadata->max_mutation_of_new_neuron_in_new_layer_chance_mutation_probability)
			* (evolution_metadata->mutation_chance_of_new_neuron_in_new_layer_chance_mutation_probability > ValueGeneration::NextDouble());

	}

	void AugmentTopology()
	{
		bool in_new_layer = evolution_metadata->new_neuron_in_new_layer_chance > ValueGeneration::NextDouble();
		
		int highest_fitness_i = -1;
		double highest_fitness = -1;
		for (size_t i = 0; i < evolution_metadata->allowed_new_neuron_IDs.size(); i++)
		{
			double current_neuron_fitness = ValueGeneration::NextDouble() * evolution_metadata->neuron_type_probabilities[i];
			bool is_highest_fitness = current_neuron_fitness > highest_fitness;

			highest_fitness_i += (i - highest_fitness_i) * is_highest_fitness;
			highest_fitness += (current_neuron_fitness - highest_fitness) * is_highest_fitness;
		}
		NeuronTypeIdentifier selected_neuron = (NeuronTypeIdentifier)evolution_metadata->allowed_new_neuron_IDs[highest_fitness_i];

		if (in_new_layer)
		{
			size_t layer_insert_i = (size_t)std::round(ValueGeneration::NextDouble() * (shape_length - 1));
		}
	}

	///<param name="insert_i">insert_i doesn't count input layer so insert_i will insert in shape[insert_i + 1]</param>
	/// <returns>starting Neuron_i of new layer</returns>
	size_t AddLayerToShape(size_t insert_i, size_t insert_layer_neuron_count = 1)
	{
		size_t new_layer_neuron_i = shape[0];
		for (size_t i = 1; i < insert_i + 1; i++)
		{
			new_layer_neuron_i += shape[i];
		}

		// Move layers
		for (int i = shape_length - 1; i >= 0; i--)
		{
			shape[i + 1] = shape[i];
		}
		shape_length += insert_layer_neuron_count;
		shape[insert_i] = insert_layer_neuron_count;

		return new_layer_neuron_i;
	}

	static const size_t metadata_value_count = 10;

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
		size_t metadata[metadata_value_count]
		{
			neuron_count,
			input_length,
			output_length,
			execution_results_value_count,
			gradients_value_count,
			shape_length,
			max_neuron_count,
			max_layer_count,
			evolution_metadata != 0,
			evolution_metadata != 0? evolution_metadata->allowed_new_neuron_IDs.size() : 0
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
			case DenseNeuronId:
				neuron_size = sizeof(DenseNeuron);
				connections_size = sizeof(DenseConnections);
				break;
			case DenseLSTMId:
				neuron_size = sizeof(DenseLSTM);
				connections_size = sizeof(DenseConnections);
				break;
			case NEATNeuronId:
				neuron_size = sizeof(NEATNeuron);
				connections_size = sizeof(NEATConnections);
				break;
			case NEATLSTMId:
				neuron_size = sizeof(NEATLSTM);
				connections_size = sizeof(NEATConnections);
				break;
			default:
				throw std::exception("NeuronType not implemented");
			}
			fwrite(current_neuron, neuron_size, 1, nn_file);
			fwrite(current_neuron->connections, connections_size, 1, nn_file);
			fwrite(current_neuron->connections->GetWeights(), sizeof(double), weight_count, nn_file);
			neurons[i]->connections->WriteNonInheritedValues(nn_file);
		}

		if (evolution_metadata != 0)
		{
			fwrite(evolution_metadata, sizeof(EvolutionMetaData), 1, nn_file);
			fwrite(evolution_metadata->allowed_new_neuron_IDs.data(), sizeof(int), evolution_metadata->allowed_new_neuron_IDs.size(), nn_file);
			fwrite(evolution_metadata->neuron_type_probabilities.data(), sizeof(double), evolution_metadata->neuron_type_probabilities.size(), nn_file);
		}

		fclose(nn_file);
	}

	static NN* Load(std::string path_with_no_extension)
	{
		size_t metadata[metadata_value_count]{};
		//neuron_count,
		//input_length,
		//output_length,
		//execution_results_value_count,
		//gradients_value_count,
		//shape_length,
		//max_neuron_count,
		//max_layer_count,
		//evolution_metadata != 0
		//evolution_allowed_neuron_ids_count

		FILE* nt_file;
		if (fopen_s(&nt_file, (path_with_no_extension + GetNeuronTypeFileExtension()).data(), "rb"))
			throw std::exception("File cannot be opened");

		fread(&metadata, sizeof(size_t), metadata_value_count, nt_file);

		size_t neuron_count = metadata[0];
		size_t input_length = metadata[1];
		size_t output_length = metadata[2];
		size_t execution_results_value_count = metadata[3];
		size_t gradients_value_count = metadata[4];
		size_t shape_length = metadata[5];
		size_t max_neuron_count = metadata[6];
		size_t max_layer_count = metadata[7];
		bool evolution_metadata_written = metadata[8];
		size_t allowed_neuron_id_count = metadata[9];

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

				break;

			case DenseLSTMId:
				neuron = (DenseLSTM*)malloc(sizeof(DenseLSTM));
				fread(neuron, sizeof(DenseLSTM), 1, nn_file);

				neuron->connections = (DenseConnections*)malloc(sizeof(DenseConnections));
				fread(neuron->connections, sizeof(DenseConnections), 1, nn_file);
				break;
			case NEATNeuronId:
				neuron = (NEATNeuron*)malloc(sizeof(NEATNeuron));
				fread(neuron, sizeof(NEATNeuron), 1, nn_file);

				neuron->connections = (NEATConnections*)malloc(sizeof(NEATConnections));
				fread(neuron->connections, sizeof(NEATConnections), 1, nn_file);

				break;
			case NEATLSTMId:
				neuron = (NEATLSTM*)malloc(sizeof(NEATLSTM));
				fread(neuron, sizeof(NEATLSTM), 1, nn_file);

				neuron->connections = (NEATConnections*)malloc(sizeof(NEATConnections));
				fread(neuron->connections, sizeof(NEATConnections), 1, nn_file);

				break;
			default:
				throw std::exception("Neuron not implemented for loading");
			}

			weight_count = neuron->connections->GetWeightCount();
			weights = new double[weight_count];
			fread(weights, sizeof(double), weight_count, nn_file);
			neuron->connections->SetWeights(weights);
			weights = 0;

			neuron->connections->ReadNonInheritedValues(nn_file);

			neurons[i] = neuron;
		}
		
		EvolutionMetaData* evolution_values = 0;
		if (evolution_metadata_written)
		{
			evolution_values = (EvolutionMetaData*)malloc(sizeof(EvolutionMetaData));
			if (evolution_values == 0)
				throw std::string("System out of memory");
			fread(evolution_values, sizeof(EvolutionMetaData), 1, nn_file);
			
			evolution_values->allowed_new_neuron_IDs = std::vector<int>();
			evolution_values->neuron_type_probabilities = std::vector<double>();

			int* allowed_neuron_ids = new int[allowed_neuron_id_count];
			fread(allowed_neuron_ids, sizeof(int), allowed_neuron_id_count, nn_file);

			double* per_neuron_probability = new double[allowed_neuron_id_count];
			fread(per_neuron_probability, sizeof(double), allowed_neuron_id_count, nn_file);

			for (size_t i = 0; i < allowed_neuron_id_count; i++)
			{
				evolution_values->allowed_new_neuron_IDs.push_back(allowed_neuron_ids[i]);
				evolution_values->neuron_type_probabilities.push_back(per_neuron_probability[i]);
			}

			delete[] allowed_neuron_ids;
			delete[] per_neuron_probability;
		}
		
		fclose(nn_file);

		NN* out = new NN(neurons, neuron_count, input_length, output_length, shape, shape_length, 0, false, neuron_types, true, max_neuron_count, max_layer_count, evolution_values, false);
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
