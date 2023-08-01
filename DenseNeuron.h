#include "INeuron.h"
#include "DenseConnections.h"
#include "ActivationFunctions.h"
#include "Derivatives.h"

#pragma once
class DenseNeuron : public INeuron
{
public:
	ActivationFunctions::ActivationFunction activation_function;

	DenseNeuron(size_t neuron_i, size_t previous_layer_start_i, size_t previous_layer_length, ActivationFunctions::ActivationFunction activation_function)
	{
		this->activation_function = activation_function;
		neuron_written_gradient_count = 2;
		connections = new DenseConnections(previous_layer_start_i, previous_layer_length, neuron_written_gradient_count);
	}

	void INeuron::ExecuteStore(double* activations, double* execution_results)
	{

	}
	
	double INeuron::Execute(double* activations)
	{

	}
	
	void INeuron::GetGradients(double* gradients, double* costs, double* execution_results)
	{

	}
	
	void INeuron::GetGradients(double* gradients, double* costs, size_t t_length, size_t t_count)
	{

	}
	
	void INeuron::SubtractGradients(double* gradients, double learning_rate)
	{

	}
	
	void INeuron::SubtractGradients(double* gradients, double learning_rate, size_t t_length, size_t t_count)
	{

	}

	void INeuron::DeleteMemory()
	{

	}

	void Free()
	{
		connections->Free();
	}
};

