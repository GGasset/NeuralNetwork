#pragma once
#include "LSTMNeuron.h"
#include "NEATConnections.h"

class NEATLSTM :
    public LSTMNeuron
{
    NEATLSTM(size_t neuron_i, size_t connections_start_i, size_t last_connection_i, double chance_of_connection = 1, int8_t weight_direction_from_0 = 0)
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

        this->connections = new NEATConnections(connections_start_i, last_connection_i, neuron_written_gradient_count, chance_of_connection, weight_direction_from_0);
    }
};

