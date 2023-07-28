#include <iostream>

#pragma once
class IConnections
{
protected:
	/// <summary>
	/// Input layer not included as its activations are the input
	/// </summary>
	size_t layer_i;
	size_t neuron_i;
	size_t weight_count;
	double* weights;
	
public:

	size_t GetLayerI()
	{
		return layer_i;
	}

	size_t GetNeuronI()
	{
		return neuron_i;
	}

	size_t GetWeightCount()
	{
		return weight_count;
	}

	void Free()
	{
		if (weights)
			free(weights);
	}

	virtual double LinearFunction(double** network_activations) = 0;
	virtual double* GetGradients(size_t output_write_start, double** network_activations, double** network_costs, double linear_function_gradient) = 0;
	virtual void SubtractGradients(double* gradients, size_t input_read_start) = 0;
	virtual void SubtractGradients(double** gradients, size_t input_read_start) = 0;
};

