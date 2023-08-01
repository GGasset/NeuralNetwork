#include "IConnections.h"
#include "ValueGeneration.h"
#include <string>

#pragma once
class DenseConnections :
    public IConnections
{
    size_t previous_layer_start_i = -1;
    size_t previous_layer_length = -1;

public:
    double IConnections::LinearFunction(double* network_activations) = 0;
    void IConnections::CalculateGradients(double* gradients, double* execution_results) = 0;
    void IConnections::SubtractGradients(double* gradients) = 0;
    void IConnections::SubtractGradients(double* gradients) = 0;
};

