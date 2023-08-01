#include <iostream>

#pragma once
#include <tuple>

class IConnections
{
protected:
	/// <summary>
	/// Input layer is included although it isn't instantiated
	/// </summary>
	size_t neuron_i = -1;

	/// <summary>
	/// Execution results must have the same values per neuron as gradients per neuron
	/// </summary>
	size_t self_gradients_start_i = -1;
	size_t self_gradients_length = -1;
	size_t neuron_written_gradient_count = -1;
	size_t weight_count = -1;
	double* weights = 0;
	
public:

	void Free()
	{
		if (weights)
			delete[] weights;
		delete this;
	}

	virtual double LinearFunction(double* network_activations) = 0;
	virtual void CalculateGradients(double* gradients, double* neuron_activations) = 0;
	virtual void SubtractGradients(double* gradients) = 0;
	virtual void SubtractGradients(double* gradients) = 0;
};

