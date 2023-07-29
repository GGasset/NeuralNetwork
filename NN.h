#include <list>
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
	double*** ExecuteStore(double* X)
	{
		double*** execution_results = new double** [shape_length - 1];
		double** network_activations = new double* [shape_length];
		network_activations[0] = X;

		std::list<ILayer*>::iterator it = layers.begin();
		for (size_t i = 1; it != layers.end() && i < shape_length; i++, it++)
		{
			size_t layer_length = shape[i];

			execution_results[i - 1] = new double* [layer_length];
			for (size_t j = 0; j < layer_length; j++)
			{
				execution_results[i - 1][j] = (*it)->neurons[j]->ExecuteStore(network_activations);
			}
		}
	}

public:
	double* Execute(double* X)
	{
		return 0;
	}

public:
	/// <summary>
	/// Also recommended for recurrent layers_not_including_input_layer, if there are none recurrent layers_not_including_input_layer this will function as a batch
	/// </summary>
	void Supervised_Train(size_t t_count, double** X, double** Y, Cost::CostFunction cost_function, size_t starting_i = 0)
	{

	}

	/// <summary>
	/// Perfect for making batches without recurrent layers_not_including_input_layer
	/// </summary>
	void Supervised_Train(double** X, double** Y, size_t step_count, size_t batch_size)
	{

	}
	
	/// <summary>
	/// Perfect for making batches with recurrent layers_not_including_input_layer
	/// </summary>
	void Supervised_Train(double*** X, double*** Y, size_t step_count, size_t* t_count_per_step, size_t batch_size = 1)
	{

	}
};

