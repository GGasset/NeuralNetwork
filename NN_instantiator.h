#include "NN.h"

#pragma once
class NN_instantiator
{
	/// <param name="shape">| Input layer isn't instantiated</param>
	/// <param name="layer_types">| Don't include input layer</param>
	/// <param name="layer_count">| Input layer is included in the length</param>
	/// <param name="layer_activations">| There are some neurons like DenseLSTM that don't need this, use None for example</param>
	static NN* Instantiate(size_t* shape, NN::NeuronTypeIdentifier* layer_types, size_t layer_count, ActivationFunctions::ActivationFunction* layer_activations, bool free_layer_types = true, bool free_layer_activations = true)
	{
		size_t input_length = shape[0];

		size_t neuron_count = 0;
		for (size_t i = 1; i < layer_count; i++)
		{
			neuron_count += shape[i];
		}

		INeuron** neurons = new INeuron * [neuron_count];
		NN::NeuronTypeIdentifier* neuron_ids = new NN::NeuronTypeIdentifier[neuron_count];

		size_t neuron_i = 0;
		size_t previous_layer_start_i = 0;
		for (size_t i = 1; i < layer_count; i++)
		{
			for (size_t j = 0; j < shape[1] && (neuron_i < neuron_count); j++)
			{
				neuron_ids[neuron_i] = layer_types[i];
				switch (layer_types[i - 1])	
				{
				case NN::DenseNeuronId:
					neurons[i] = new DenseNeuron(neuron_i, previous_layer_start_i, shape[i - 1], layer_activations[i]);
					break;
				case NN::DenseLSTMId:
					neurons[i] = new DenseLSTM(neuron_i, previous_layer_start_i, shape[i - 1]);
					break;
				default:
					throw std::exception("Neuron type not implemented for automatic instantiation");
				}
			}
			previous_layer_start_i += shape[i - 1];
		}
		delete[] neuron_ids;
		if (free_layer_activations)
			delete[] layer_activations;
		if (free_layer_types)
			delete[] layer_types;

		NN* out = new NN(neurons, neuron_count, input_length, shape[layer_count - 1], neuron_ids);
		return out;
	}
};

