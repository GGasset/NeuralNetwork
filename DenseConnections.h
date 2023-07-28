#include "IConnections.h"
#include "ValueGeneration.h"
#include <string>

#pragma once
class DenseConnections :
    public IConnections
{
public:
    DenseConnections(size_t layer_i, size_t neuron_i, size_t* network_shape)
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

    double IConnections::LinearFunction(double** network_activations)
    {
        double output = 0;
        double* previous_layer_activations = network_activations[layer_i - 1];
        for (size_t i = 0; i < weight_count; i++)
        {
            output += previous_layer_activations[i] * weights[i];
        };
        return output;
    }

    void IConnections::GetGradients(size_t output_write_start, double* output, double** network_activations, double** network_costs, double linear_function_gradient)
    {
        double* previous_layer_costs = network_costs[layer_i - 1];
        double* previous_layer_activations = network_activations[layer_i - 1];
        for (size_t i = 0; i < weight_count; i++)
        {
            output[i + output_write_start] = linear_function_gradient * previous_layer_activations[i];
            previous_layer_costs[i] = -linear_function_gradient * weights[i];
        }
    }

    void IConnections::SubtractGradients(double* gradients, size_t input_read_start, double learning_rate)
    {
        for (size_t i = 0; i < weight_count; i++)
        {
            weights[i] -= gradients[input_read_start + i] * learning_rate;
        }
    }

    void IConnections::SubtractGradients(double**** network_gradients, size_t t_count, size_t input_read_start, double learning_rate)
    {
        for (size_t t = 0; t < t_count; t++)
        {
            SubtractGradients(network_gradients[t][layer_i][neuron_i], input_read_start, learning_rate);
        }
    }
};

