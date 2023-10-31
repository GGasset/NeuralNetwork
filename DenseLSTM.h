#include "DenseConnections.h"
#include "LSTMNeuron.h"

#pragma once
class DenseLSTM : public LSTMNeuron
{
public:
	/// <param name="weight_direction_from_0">
	/// direction -1: exclusively negative |  
	/// direction 0: not exclusive direction | 
	/// direction 1: exclusively positive
	/// </param>
	DenseLSTM(size_t neuron_i, size_t previous_layer_start_i, size_t previous_layer_length, int8_t weight_direction_from_0 = 0)
	{
		this->neuron_i = neuron_i;

		neuron_written_gradient_count = 6;
		neuron_written_execution_results_count = 10;

		double min_value = -2.0 * ((weight_direction_from_0 < 0) || !weight_direction_from_0);
		double max_value = 2.0 * ((weight_direction_from_0 > 0) || !weight_direction_from_0);
		forget_weight = ValueGeneration::GenerateWeight(min_value, 0.5, max_value);
		sigmoid_store_weight = ValueGeneration::GenerateWeight(min_value, 0.5, max_value);
		tanh_store_weight = ValueGeneration::GenerateWeight(min_value, 0.5, max_value);
		output_weight = ValueGeneration::GenerateWeight(min_value, 0.5, max_value);

		this->connections = new DenseConnections(previous_layer_start_i, previous_layer_length, neuron_written_gradient_count, weight_direction_from_0);
	}
};
