#include "DenseConnections.h"
#include "Neuron.h"
#include "ILayer.h"

#pragma once
class DenseNeuronLayer : public ILayer
{
public:
	DenseNeuronLayer(size_t* network_shape, size_t layer_i, ActivationFunctions::ActivationFunction activation_function)
	{
		this->layer_length = network_shape[layer_i];
		this->neurons = new INeuron*[layer_length];
		for (size_t i = 0; i < layer_length; i++)
		{
			DenseConnections* connections = new DenseConnections(layer_i, i, network_shape);
			Neuron* neuron = new Neuron(connections, 1, activation_function);
			this->neurons[i] = neuron;
		}
	}
};

