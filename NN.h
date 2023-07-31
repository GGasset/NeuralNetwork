#include <list>
#include <tuple>
#include "LayerLibrary.h"

#pragma once
class NN
{
private:
	ILayer** layers = 0;
	size_t* shape = 0;
	size_t shape_length = -1;

public:
	NN(ILayer** layers_not_including_input_layer, size_t* shape_including_input_layer, size_t shape_length)
	{
		this->layers = layers_not_including_input_layer;
		this->shape = shape_including_input_layer;

		this->shape_length = shape_length;
	}

private:
	std::tuple<double***, double**> ExecuteStore(double* X)
	{
		double*** execution_results = new double** [shape_length];
		double** network_activations = new double* [shape_length];
		network_activations[0] = X;
		execution_results[0] = NULL;

		for (size_t i = 1; i < shape_length; i++)
		{
			size_t layer_length = shape[i];
			ILayer* current_layer = layers[i - 1];

			double* current_layer_network_activations;
			current_layer_network_activations = network_activations[i] = new double[layer_length];

			double** current_layer_execution_results;
			current_layer_execution_results = execution_results[i] = new double* [layer_length];
			for (size_t j = 0; j < layer_length; j++)
			{
				INeuron* current_neuron = current_layer->neurons[j];
				double* output;
				output = current_layer_execution_results[j] = current_neuron->ExecuteStore(network_activations);
				current_layer_network_activations[j] = current_neuron->GetOutput(output);
			}
		}

		return std::tuple<double***, double**>(execution_results, network_activations);
	}

public:
	double* Execute(double* X)
	{
		double** network_activations = new double* [shape_length];
		network_activations[0] = X;

		for (size_t i = 1; i < shape_length; i++)
		{
			size_t layer_length = shape[i];
			ILayer* current_layer = layers[i - 1];

			double* current_layer_network_activations;
			current_layer_network_activations = network_activations[i] = new double[layer_length];
			for (size_t j = 0; j < layer_length; j++)
			{
				INeuron* current_neuron = current_layer->neurons[j];

				network_activations[i][j] = current_neuron->Execute(network_activations);
			}
		}

		for (size_t i = 1; i < shape_length - 1; i++)
		{
			delete[] network_activations[i];
		}
		double* output = network_activations[shape_length - 1];
		delete[] network_activations;

		return output;
	}

private:
	std::tuple<double****, double***> allocate_gradients_and_costs(size_t t_count, double** output_costs)
	{
		double**** gradients = new double*** [t_count];
		double*** network_costs = new double** [t_count];

		for (size_t t = 0; t < t_count; t++)
		{
			gradients[t] = new double** [shape_length];
			network_costs[t] = new double* [shape_length];
			for (size_t layer_i = 0; layer_i < shape_length - 1; layer_i++)
			{
				size_t layer_length = shape[layer_i];
				gradients[t][layer_i] = new double* [layer_length];
				network_costs[t][layer_i] = new double[layer_length];
				for (size_t neuron_i = 0; neuron_i < shape[layer_i]; neuron_i++)
				{
					network_costs[t][layer_i][neuron_i] = 0.0;
				}
			}

			network_costs[t][shape_length - 1] = output_costs[t];
			gradients[t][shape_length - 1] = new double* [shape[shape_length - 1]];
		}
		delete[] output_costs;

		return std::tuple<double****, double***>(gradients, network_costs);
	}

	void deallocate_gradients_and_costs(size_t t_count, double**** gradients, double*** network_costs)
	{
		for (size_t t = 0; t < t_count; t++)
		{
			for (size_t layer_i = 0; layer_i < shape_length; layer_i++)
			{
				for (size_t neuron_i = 0; neuron_i < shape[layer_i] && layer_i > 0; neuron_i++)
				{
					delete[] gradients[t][layer_i][neuron_i];
				}
				delete[] gradients[t][layer_i];
				delete[] network_costs[t][layer_i];
			}
			delete[] gradients[t];
			delete[] network_costs[t];
		}

		delete[] gradients;
		delete[] network_costs;
	}

public:
	/// <summary>
	/// Also recommended for recurrent layers_not_including_input_layer, if there are none recurrent layers_not_including_input_layer this will function as a batch
	/// </summary>
	void Supervised_Train(size_t t_count, double** X, double** Y, Cost::CostFunction cost_function, double learning_rate, size_t starting_i = 0)
	{
		size_t real_t_count = t_count - starting_i;
		double**** execution_results = new double*** [real_t_count];
		double*** network_activations = new double** [real_t_count];
		double** output_costs = new double* [real_t_count];
		for (size_t t = 0; t < real_t_count; t++)
		{
			std::tuple<double***, double**> inference_execution_results = ExecuteStore(X[t]);

			execution_results[t] = std::get<0>(inference_execution_results);
			network_activations[t] = std::get<1>(inference_execution_results);

			double* network_output = network_activations[t][shape_length - 1];
			output_costs[t] = Derivatives::DerivativeOf(shape[shape_length - 1], network_output, Y[t], cost_function);
		}

		auto training_allocation = allocate_gradients_and_costs(real_t_count, output_costs);
		double**** gradients = std::get<0>(training_allocation);
		double*** network_costs = std::get<1>(training_allocation);
		
		for (int layer_i = shape_length - 1; layer_i >= 1; layer_i--)
		{
			// current layer having not allocated neurons when layer_i == 2 - layers[1] == NULL
			size_t layer_length = shape[layer_i];
			ILayer* current_layer = layers[layer_i - 1];
			if (layer_i == 2)
				int x = 0;
			for (size_t i = 0; i < layer_length; i++)
			{
				INeuron* current_neuron = current_layer->neurons[i];

				current_neuron->GetGradients(execution_results, real_t_count, gradients, network_costs, network_activations);
			}
		}

		// Subtract Gradients
		for (int layer_i = 1; layer_i < shape_length; layer_i++)
		{
			size_t layer_length = shape[layer_i];
			ILayer* current_layer = layers[layer_i - 1];

			for (size_t k = 0; k < layer_length; k++)
			{
				INeuron* current_neuron = current_layer->neurons[k];

				current_neuron->SubtractGradients(gradients, real_t_count, learning_rate);
			}
		}

		deallocate_gradients_and_costs(real_t_count, gradients, network_costs);
		for (size_t t = 0; t < real_t_count; t++)
		{
			/*
				Layer_i starts at 1 because:
					network_activations[t][0] is external data
					execution_results[t][0] isn't instantiated
			*/
			for (size_t layer_i = 1; layer_i < shape_length; layer_i++)
			{
				delete[] network_activations[t][layer_i];
				delete[] execution_results[t][layer_i];
			}
			delete[] network_activations[t];
			delete[] execution_results[t];
		}
		delete[] network_activations;
		delete[] execution_results;
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
		for (size_t i = 0; i < shape_length - 1; i++)
		{
			layers[i]->free();
		}
		delete layers;
		delete[] shape;
	}
};
