#pragma once
#include "IConnections.h"
#include <vector>

class NEATConnections :
	public IConnections
{
protected:
	std::vector<size_t> connections_indices;

public:
	double IConnections::LinearFunction(double* network_activations, size_t t_index = 0)
	{
		size_t activations_t_addition = t_index * network_neuron_count;
		double linear_function = 0;
		for (size_t i = 0; i < weight_count; i++)
			linear_function += network_activations[activations_t_addition + connections_indices[i]] * weights[i];

		return linear_function;
	}

	void IConnections::CalculateGradients(double* gradients, double* neuron_activations, double* costs, double* linear_function_gradients, size_t t_count)
	{
		for (size_t t = 0; t < t_count; t++)
		{
			double current_linear_function_gradient = linear_function_gradients[t];

			size_t self_gradients_start_i = network_gradients_value_count * t + this->self_gradients_start_i + neuron_written_gradient_count;
			size_t first_neuron_i = network_neuron_count * t;
			for (size_t i = 0; i < weight_count; i++)
			{
				size_t connected_neuron_i = first_neuron_i + connections_indices[i];
				size_t gradient_i = self_gradients_start_i + i;

				gradients[self_gradients_start_i + i] = current_linear_function_gradient * neuron_activations[connected_neuron_i];
				costs[connected_neuron_i] -= current_linear_function_gradient * weights[i];
			}
		}
	}

	virtual double CalculateDerivative(double* network_activations, size_t t_index)
	{
		double derivative = 0;

		size_t current_t_activations_start_i = network_neuron_count * t_index;
		for (size_t i = 0; i < weight_count; i++)
			derivative += network_activations[current_t_activations_start_i + connections_indices[i]] + weights[i];
		
		return derivative;
	}


	void IConnections::SubtractGradients(double* gradients, size_t t_count, double learning_rate)
	{
		for (size_t t = 0; t < t_count; t++)
		{
			size_t gradients_start_i = t * network_gradients_value_count + self_gradients_start_i + neuron_written_gradient_count;

			for (size_t i = 0; i < weight_count; i++)
				weights[i] -= gradients[gradients_start_i + i] * learning_rate;
		}
	}

	/// <summary>
	/// TODO: Set network assigned values again to all neurons
	/// </summary>
	void AdjustToNewNeuron(size_t insert_i, bool add_connection, int8_t weight_direction_from_0 = 0) override
	{
		for (size_t i = 0; i < connections_indices.size(); i++)
			connections_indices[i] += connections_indices[i] >= insert_i;

		if (!add_connection)
			return;

		connections_indices.push_back(insert_i);

		// Add weight
		double* new_weights = new double[weight_count + 1];
		for (size_t i = 0; i < weight_count; i++)
			new_weights[i] = weights[i];

		double weight_range = 1 / sqrt(weight_count + 1);
		double min_value = -weight_range * ((weight_direction_from_0 < 0) || !weight_direction_from_0);
		double max_value = weight_range * ((weight_direction_from_0 > 0) || !weight_direction_from_0);

		double new_weight = ValueGeneration::GenerateWeight(min_value, 0, max_value);
		new_weights[weight_count] = new_weight;
		
		delete[] weights;
		SetWeights(new_weights);
		weight_count++;
	}

	void AdjustToDeletedNeuron(size_t deleted_i) override
	{
		size_t connection_i = -1;
		for (size_t i = 0; i < connections_indices.size() && connection_i != -1; i++)
		{
			connections_indices[i] -= connections_indices[i] >= deleted_i;
			connection_i += (i - connection_i) * (connections_indices[i] == deleted_i);
		}

		if (connection_i == -1)
			return;

		connections_indices.erase(connections_indices.begin() + connection_i);

		// Delete weight
		double* new_weights = new double[weight_count - 1];
		for (size_t i = 0; i < weight_count - 1; i++)
		{
			if (i == connection_i)
				continue;

			new_weights[i - (i > connection_i)] = weights[i];
		}

		delete[] weights;
		SetWeights(new_weights);
		weight_count--;
	}
};
