#include "IConnections.h"

#pragma once
class INeuron
{
protected:
	size_t neuron_i = -1;
	size_t neuron_written_gradient_count = -1;

	double bias = 1;

public:
	size_t GetNeuronWrittenGradientCount()
	{
		return neuron_written_gradient_count;
	}

	/// <summary>
	/// Execution results must have the same values per neuron as gradients per neuron.
	/// Value will be automatically asigned by NN
	/// </summary>
	size_t self_execution_results_start_i = -1;

	IConnections* connections = 0;

	/// <summary>
	/// Modify activations and execution results
	/// </summary>
	virtual void ExecuteStore(double* activations, double* execution_results) = 0;

	/// <summary>
	/// Modify activations
	/// </summary>
	virtual double Execute(double* activations) = 0;
	virtual void GetGradients(double* gradients, double* costs, double* execution_results) = 0;
	virtual void GetGradients(double* gradients, double* costs, size_t t_length, size_t t_count) = 0;
	virtual void SubtractGradients(double* gradients, double learning_rate) = 0;
	virtual void SubtractGradients(double* gradients, double learning_rate, size_t t_length, size_t t_count) = 0;
	virtual void DeleteMemory() = 0;
	virtual void Free() = 0;
};

