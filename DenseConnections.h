#include "IConnections.h"
#include "ValueGeneration.h"
#include <string>

#pragma once
class DenseConnections :
    public IConnections
{
public:
    DenseConnections(size_t layer_i, size_t neuron_i, int* network_shape)
    {
        this->layer_i = layer_i;
        this->neuron_i = neuron_i;
        
        size_t previous_layer_length = network_shape[layer_i];
        this->weight_count = previous_layer_length;

        this->weights = new double[previous_layer_length];
        if (!this->weights)
            throw std::string("Out of memory");
        for (size_t i = 0; i < previous_layer_length; i++)
        {
            this->weights[i] = ValueGeneration::GenerateWeight(-2, 0.5, 2);
        }
    }
};

