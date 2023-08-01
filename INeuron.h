#include "IConnections.h"

#pragma once
class INeuron
{
protected:
	size_t neuron_i = -1;

	double bias = 0;

public:
	/// <summary>
	/// Execution results must have the same values per neuron as gradients per neuron.
	/// Value will be automatically asigned by NN
	/// </summary>
	size_t self_execution_results_start_i = -1;
	/// <summary>
	/// Value will be automatically asigned by NN
	/// </summary>
	size_t self_execution_results_length = -1;

	IConnections* connections = 0;


	virtual void ExecuteStore(double* activations, double* execution_results) = 0;
	virtual void Execute(double* activations) = 0;
	virtual void GetGradients(double* gradients, double* costs, double* execution_results) = 0;
	virtual void GetGradients(double* gradients, double* costs, size_t t_length, size_t t_count) = 0;
	virtual void SubtractGradients(double* gradients, double learning_rate) = 0;
	virtual void SubtractGradients(double* gradients, double learning_rate, size_t t_length, size_t t_count) = 0;
	virtual void DeleteMemory() = 0;
	virtual void Free() = 0;
};

