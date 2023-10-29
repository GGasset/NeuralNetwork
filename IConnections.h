#include "ValueGeneration.h"
#include <stdint.h>

#pragma once

/// <summary>
/// You must implement where the connections are connected when using this class as base
/// </summary>
class IConnections
{
protected:
	size_t neuron_written_gradient_count = -1;
	size_t weight_count = -1;
	double* weights = 0;

	/// <param name="weight_direction_from_0">
	/// direction -1: exclusively negative |  
	/// direction 0: not exclusive direction | 
	/// direction 1: exclusively positive
	/// </param>
	void GenerateWeights(int8_t weight_direction_from_0 = 0)
	{
		if (weights)
		{
			delete[] weights;
			weights = 0;
		}

		weights = new double[weight_count];
		double weight_range = 1 / sqrt(weight_count);
		double min_value = -weight_range * ((weight_direction_from_0 < 0) || !weight_direction_from_0);
		double max_value = weight_range * ((weight_direction_from_0 > 0) || !weight_direction_from_0);
		for (size_t i = 0; i < weight_count; i++)
		{
			weights[i] = ValueGeneration::GenerateWeight(min_value, 0, max_value);
		}
	}

public:

	/// <summary>
	/// Execution results must have the same values per neuron as gradients per neuron
	/// Value intialized by NN
	/// </summary>
	size_t self_gradients_start_i = -1;


	/// <summary>
	/// Value will be automatically initialized by NN and is made to be accessed by neuron
	/// </summary>
	size_t network_execution_results_value_count = -1;

	/// <summary>
	/// Value will be automatically initialized by NN
	/// </summary>
	size_t network_gradients_value_count = -1;

	/// <summary>
	/// Value will be automatically initialized by NN
	/// </summary>
	size_t network_neuron_count = -1;

	size_t GetWeightCount()
	{
		return weight_count;
	}

	double* GetWeights()
	{
		return weights;
	}

	void SetWeights(double* weights)
	{
		this->weights = weights;
	}

	void Free()
	{
		if (weights)
			delete[] weights;
		delete this;
	}

	virtual void AdjustToNewNeuron(size_t insert_i, bool add_connection, int8_t weight_direction_from_0 = 0)
	{

	}

	virtual void AdjustToDeletedNeuron(size_t deleted_i)
	{

	}

	virtual double LinearFunction(double* network_activations, size_t t_index = 0) = 0;
	virtual double CalculateDerivative(double* network_activations, size_t t_index) = 0;
	virtual void CalculateGradients(double* gradients, double* neuron_activations, double* costs, double* linear_function_gradients, size_t t_count) = 0;
	virtual void SubtractGradients(double* gradients, size_t t_count, double learning_rate) = 0;
};

