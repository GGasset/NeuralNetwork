#include "DenseConnections.h"
#include "INeuron.h"

#pragma once
class DenseLSTM : public INeuron
{
protected:
	double forget_weight;
	double sigmoid_store_weight;
	double tanh_store_weight;
	double output_weight;

public:
	DenseLSTM(size_t neuron_i, size_t previous_layer_start_i, size_t previous_layer_length)
	{
		this->neuron_i = neuron_i;

		neuron_written_gradient_count = 10E5;

		forget_weight = ValueGeneration::GenerateWeight(-2, 0.5, 2);
		sigmoid_store_weight = ValueGeneration::GenerateWeight(-2, 0.5, 2);
		tanh_store_weight = ValueGeneration::GenerateWeight(-2, 0.5, 2);
		output_weight = ValueGeneration::GenerateWeight(-2, 0.5, 2);

		this->connections = new DenseConnections(previous_layer_start_i, previous_layer_length, neuron_written_gradient_count);
	}
};

