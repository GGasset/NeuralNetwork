#pragma once
#include "NeuronBase.h"
#include "NEATConnections.h"
class NEATNeuron :
    public NeuronBase
{
    NEATNeuron(size_t neuron_i, size_t connections_start_i, size_t last_connection_i, ActivationFunctions::ActivationFunction activation, 
        double chance_of_connection = 1, int8_t weight_direction_from_0 = 0)
    {
        this->neuron_i = neuron_i;
        this->activation_function = activation;

        neuron_written_gradient_count = 1;
        neuron_written_execution_results_count = 1;

        this->connections = new NEATConnections(connections_start_i, last_connection_i, neuron_written_gradient_count,
            chance_of_connection, weight_direction_from_0);
    }
};

