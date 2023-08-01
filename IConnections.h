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
	size_t self_execution_results_start_i = -1;
	size_t self_execution_results_length = -1;
	size_t neuron_gradients_written_value_count = -1;
	size_t weight_count = -1;
	double* weights = 0;
	
public:

	size_t GetNeuronI()
	{
		return neuron_i;
	}

	size_t GetExecutionResultsStart()
	{
		return self_execution_results_start_i;
	}

	size_t GetExecutionResultsLength()
	{
		return self_execution_results_length;
	}

	void Free()
	{
		if (weights)
			delete[] weights;
		delete this;
	}

	virtual double LinearFunction(double* network_activations) = 0;
	virtual void CalculateGradients(double* gradients, double* execution_results) = 0;
	virtual void SubtractGradients(double* gradients) = 0;
	virtual void SubtractGradients(double* gradients) = 0;
};

