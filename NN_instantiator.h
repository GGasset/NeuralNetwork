#include "NN.h"

#pragma once
class NN_instantiator
{
	/// <param name="shape">| Input layer isn't instantiated</param>
	/// <param name="layer_types">| Don't include input layer</param>
	/// <param name="layer_count">| Input layer is included in the length</param>
	/// <param name="layer_activations">| There are some neurons like DenseLSTM that don't need this, use None for example</param>
	static NN* Instantiate(size_t* shape, NN::NeuronTypeIdentifier* layer_types, size_t layer_count, ActivationFunctions::ActivationFunction* layer_activations, int8_t weight_direction_from_0 = 0, size_t max_neuron_count = 0, EvolutionMetaData* evolution_values = 0, bool free_layer_types = true, bool free_layer_activations = true)
	{
		int8_t* layer_weight_direction = new int8_t[layer_count];
		for (size_t i = 0; i < layer_count; i++)
		{
			layer_weight_direction[i] = weight_direction_from_0;
		}
		return Instantiate(shape, layer_types, layer_count, layer_activations, layer_weight_direction, max_neuron_count, evolution_values, free_layer_types, free_layer_activations);
	}


	/// <param name="shape">| Input layer isn't instantiated</param>
	/// <param name="layer_types">| Don't include input layer</param>
	/// <param name="layer_count">| Input layer is included in the length</param>
	/// <param name="layer_activations">| There are some neurons like DenseLSTM that don't need this, use None for example</param>
	static NN* Instantiate(size_t* shape, NN::NeuronTypeIdentifier* layer_types, size_t layer_count, ActivationFunctions::ActivationFunction* layer_activations, int8_t* layer_weight_direction_from_0, size_t max_neuron_count = 0, EvolutionMetaData* evolution_values = 0, bool free_layer_types = true, bool free_layer_activations = true)
	{
		size_t input_length = shape[0];

		size_t neuron_count = 0;
		for (size_t i = 1; i < layer_count; i++)
		{
			neuron_count += shape[i];
		}

		max_neuron_count += (neuron_count - max_neuron_count) * (max_neuron_count < neuron_count);
		INeuron** neurons = new INeuron * [max_neuron_count];
		NN::NeuronTypeIdentifier* neuron_ids = new NN::NeuronTypeIdentifier[max_neuron_count];
		for (size_t i = 0; i < max_neuron_count; i++)
			neuron_ids[i] = NN::DenseNeuronId;

		size_t neuron_i = 0;
		size_t previous_layer_start_i = 0;
		for (size_t i = 1; i < layer_count; i++)
		{
			for (size_t j = 0; j < shape[1] && (neuron_i < max_neuron_count); j++, neuron_i++)
			{
				neuron_ids[neuron_i] = layer_types[i];
				switch (layer_types[i - 1])	
				{
				case NN::DenseNeuronId:
					neurons[neuron_i] = new DenseNeuron(neuron_i, previous_layer_start_i, shape[i - 1], layer_activations[i], layer_weight_direction_from_0[i]);
					break;
				case NN::DenseLSTMId:
					neurons[neuron_i] = new DenseLSTM(neuron_i, previous_layer_start_i, shape[i - 1], layer_weight_direction_from_0[i]);
					break;
				case NN::NEATNeuronId:
					neurons[neuron_i] = new NEATNeuron(neuron_i, previous_layer_start_i, previous_layer_start_i + shape[i - 1] - 1, layer_activations[i], 1, layer_weight_direction_from_0[i]);
					break;
				case NN::NEATLSTMId:
					neurons[neuron_i] = new NEATLSTM(neuron_i, previous_layer_start_i, previous_layer_start_i + shape[i - 1] - 1, 1, layer_weight_direction_from_0[i]);
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

		NN* out = new NN(neurons, neuron_count, input_length, shape[layer_count - 1], shape, layer_count, neuron_ids, true, 0, max_neuron_count);
		return out;
	}
};

