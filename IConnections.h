#include <iostream>

#pragma once
#include <tuple>

/// <summary>
/// You must implement where the connections are connected when using this class as base
/// </summary>
class IConnections
{
protected:
	/// <summary>
	/// Execution results must have the same values per neuron as gradients per neuron
	/// </summary>
	size_t self_gradients_start_i = -1;
	size_t self_gradients_length = -1;
	size_t neuron_written_gradient_count = -1;
	size_t weight_count = -1;
	double* weights = 0;
	
public:
	size_t GetWeightCount()
	{
		return weight_count;
	}

	void Free()
	{
		if (weights)
			delete[] weights;
		delete this;
	}

	virtual double LinearFunction(double* network_activations) = 0;
	virtual void CalculateGradients(double* gradients, double* neuron_activations, double* costs) = 0;
	virtual void SubtractGradients(double* gradients) = 0;
};

