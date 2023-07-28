#include <iostream>

#pragma once
class IConnections
{
protected:
	/// <summary>
	/// Input layer is included although it isn't instantiated
	/// </summary>
	size_t layer_i;
	size_t neuron_i;
	size_t weight_count;
	double* weights;
	
public:

	size_t GetLayerI()
	{
		return layer_i;
	}

	size_t GetNeuronI()
	{
		return neuron_i;
	}

	size_t GetWeightCount()
	{
		return weight_count;
	}

	void Free()
	{
		if (weights)
			free(weights);
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="network_activations">Must include input neurons activations altough they aren't instantiated as a layer</param>
	/// <returns></returns>
	virtual double LinearFunction(double** network_activations) = 0;
	virtual void GetGradients(size_t output_write_start, double* output, double** network_activations, double** network_costs, double linear_function_gradient) = 0;
	
	// TODO: add LearningRate
	virtual void SubtractGradients(double* gradients, size_t input_read_start, double learning_rate) = 0;

	/// <summary>
	/// 
	/// </summary>
	/// <param name="network_gradients_over_t">
	///		Fourth dimension: t.
	///		Third-second dimension layer-neuron.
	///		First dimension: gradients calculated at GetGradients.
	/// </param>
	virtual void SubtractGradients(double**** network_gradients_over_t, size_t input_read_start, double learning_rate) = 0;
};

