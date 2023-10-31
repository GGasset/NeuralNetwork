#include "NeuronBase.h"
#include "DenseConnections.h"

#pragma once
class DenseNeuron : public NeuronBase
{
public:
	/// <param name="weight_direction_from_0">
	/// direction -1: exclusively negative |  
	/// direction 0: not exclusive direction | 
	/// direction 1: exclusively positive
	/// </param>
	DenseNeuron(size_t neuron_i, size_t previous_layer_start_i, size_t previous_layer_length, ActivationFunctions::ActivationFunction activation_function, 
		int8_t weight_direction_from_0 = 0)
	{
		this->neuron_i = neuron_i;
		this->activation_function = activation_function;

		connectionsId = IConnections::DenseId;
		neuron_written_gradient_count = 1;
		neuron_written_execution_results_count = 1;
		connections = new DenseConnections(previous_layer_start_i, previous_layer_length, neuron_written_gradient_count, weight_direction_from_0);
	}
};

