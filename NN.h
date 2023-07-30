#include <list>
#include <tuple>
#include "LayerLibrary.h"

#pragma once
class NN
{
private:
	std::list<ILayer*> layers;
	size_t* shape = 0;
	size_t shape_length = -1;

public:
	NN(std::list<ILayer*> layers_not_including_input_layer, size_t* shape_including_input_layer)
	{
		this->layers = layers_not_including_input_layer;
		this->shape = shape_including_input_layer;

		std::list<ILayer*>::iterator it;
		shape_length = 0;
		for (it = layers_not_including_input_layer.begin(); it != layers_not_including_input_layer.end(); it++)
		{
			shape_length++;
		}
		shape_length++;
	}

private:
	std::tuple<double***, double**> ExecuteStore(double* X)
	{
		double*** execution_results = new double** [shape_length - 1];
		double** network_activations = new double* [shape_length];
		network_activations[0] = X;

		std::list<ILayer*>::iterator it = layers.begin();
		for (size_t i = 1; it != layers.end() && i < shape_length; i++, it++)
		{
			size_t layer_length = shape[i];

			double* current_layer_network_activations;
			current_layer_network_activations = network_activations[i] = new double[layer_length];

			double** current_layer_execution_results;
			current_layer_execution_results = execution_results[i - 1] = new double* [layer_length];
			for (size_t j = 0; j < layer_length; j++)
			{
				INeuron* current_neuron = (*it)->neurons[j];
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

		std::list<ILayer*>::iterator it = layers.begin();
		for (size_t i = 1; it != layers.end() && i < shape_length; i++, it++)
		{
			size_t layer_length = shape[i];

			double* current_layer_network_activations;
			current_layer_network_activations = network_activations[i] = new double[layer_length];			
			for (size_t j = 0; j < layer_length; j++)
			{
				INeuron* current_neuron = (*it)->neurons[j];
				
				network_activations[i][j] = current_neuron->Execute(network_activations);
			}
		}
	}

private:
	std::tuple<double****, double***> allocate_gradients_and_costs(size_t t_count)
	{
		double**** gradients = new double*** [t_count];
		double*** network_costs = new double** [t_count];
		for (size_t j = 0; j < shape_length; j++)
		{
			size_t layer_length = shape[j + 1];
			for (size_t t = 0; t < t_count; t++)
			{
				gradients[t] = new double** [shape_length];
				network_costs[t] = new double* [shape_length];

				gradients[t][j] = new double* [layer_length];
				network_costs[t][j] = new double[layer_length];
			}
		}

		return std::tuple<double****, double***>(gradients, network_costs);
	}

	void deallocate_gradients_and_costs(size_t t_count, double**** gradients, double*** network_costs)
	{
		for (size_t j = 0; j < shape_length; j++)
		{
			size_t layer_length = shape[j + 1];
			for (size_t t = 0; t < t_count; t++)
			{
				for (size_t k = 0; k < layer_length; k++)
				{
					delete[] gradients[t][j][k];
				}
				delete[] gradients[t][j];
				delete[] network_costs[t][j];

				delete[] gradients[t];
				delete[] network_costs[t];
			}
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
		for (size_t t = 0; t < real_t_count; t++)
		{
			std::tuple<double***, double**> inference_execution_results = ExecuteStore(X[t]);

			execution_results[t] = std::get<0>(inference_execution_results);
			network_activations[t] = std::get<1>(inference_execution_results);
		}

		auto training_allocation = allocate_gradients_and_costs(real_t_count);
		double**** gradients = std::get<0>(training_allocation);
		double*** network_costs = std::get<1>(training_allocation);
		std::list<ILayer*>::iterator it;
		int j = 0;
		for (it = layers.begin(); it != layers.end(); it++, j++)
		{
			size_t layer_length = shape[j + 1];

			for (size_t i = 0; i < layer_length; i++)
			{
				INeuron* current_neuron = (*it)->neurons[i];

				current_neuron->GetGradients(execution_results, real_t_count, gradients, network_costs, network_activations);
			}
		}
		
		// Subtract Gradients
		j = 0;
		for (it = layers.begin(); it != layers.end(); it++, j++)
		{
			size_t layer_length = shape[j + 1];
			for (size_t k = 0; k < layer_length; k++)
			{
				INeuron* current_neuron = (*it)->neurons[k];

				current_neuron->SubtractGradients(gradients, real_t_count, learning_rate);
			}
		}

		deallocate_gradients_and_costs(real_t_count, gradients, network_costs);
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
		std::list<ILayer*>::iterator it;
		for (it = layers.begin(); it != layers.end(); it++)
		{
			(*it)->free();
		}
	}
};

