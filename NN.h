#include <list>
#include <tuple>

#pragma once
class NN
{
private:

public:
	NN()
	{
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
