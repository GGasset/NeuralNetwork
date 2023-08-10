#include "DenseConnections.h"
#include "INeuron.h"
#include "Derivatives.h"

#pragma once
class DenseLSTM : public INeuron
{
protected:
	double forget_weight;
	double sigmoid_store_weight;
	double tanh_store_weight;
	double output_weight;

	double hidden_state = 0;
	double cell_state = 0;

	double first_hidden_gradient = 0;
	double first_cell_gradient = 0;

	/// <summary>
	/// Modifies cell state and hidden_state
	/// </summary>
	/// <param name="linear_hidden_adition"></param>
	/// <returns></returns>
	double calculate_forget_gate(double linear_hidden_adition)
	{
		return -1;
	}

public:
	DenseLSTM(size_t neuron_i, size_t previous_layer_start_i, size_t previous_layer_length)
	{
		this->neuron_i = neuron_i;

		neuron_written_gradient_count = 10E5;

		forget_weight = ValueGeneration::GenerateWeight(-2, 0.5, 2);
		sigmoid_store_weight = ValueGeneration::GenerateWeight(-2, 0.5, 2);
		tanh_store_weight = ValueGeneration::GenerateWeight(-2, 0.5, 2);
		output_weight = ValueGeneration::GenerateWeight(-2, 0.5, 2);

		this->connections = new DenseConnections(previous_layer_start_i, previous_layer_length, neuron_written_gradient_count);
	}

	void INeuron::ExecuteStore(double* activations, double* execution_results, size_t t_index = 0)
	{
		// execution results Structure
		// Relative indexing
		// execution_results[0] = linear_hidden_addition
		// execution_results[1] = forget sigmoid output
		// execution_results[0] = 
		// execution_results[0] = 
		


		size_t execution_results_start = t_index * connections->network_execution_results_value_count + self_execution_results_start_i;
		double linear_function = execution_results[execution_results_start] = connections->LinearFunction(activations, t_index) + bias;

	}

	void INeuron::GetGradients(double* gradients, double* costs, double* execution_results, double* network_activations, size_t t_count)
	{
		// Gradient positions
		// Relative indexing*
		// gradients[0] = output_hiddden_state
		// gradients[1] = output_cell_state_gradient
		// gradients[2] = 
		// gradients[2] = 
		// gradients[2] = 
		// gradients[2] = 
		// gradients[2] = 
		// gradients[2] = 
		// gradients[2] = 
		// gradients[2] = 
		for (size_t t = 0; t < t_count; t++)
		{
			size_t current_execution_result_start = connections->network_execution_results_value_count * t_count + self_execution_results_start_i;
			size_t previous_gradient_start = connections->network_gradients_value_count * (t_count - 1) + self_gradients_start_i;
			
			double prev_hidden_gradient = t == 0 ? first_hidden_gradient : gradients[previous_gradient_start];
			double prev_cell_gradient = t == 0 ? first_cell_gradient : gradients[previous_gradient_start + 1];
			

			double linear_hidden_addition_derivative = gradients[previous_gradient_start] * prev_hidden_gradient;
			double forget_sigmoid_derivative = linear_hidden_addition_derivative *  Derivatives::SigmoidDerivative(execution_results[current_execution_result_start]);
			double forget_weight_multiplication_derivative = execution_results[current_execution_result_start + 1] + forget_sigmoid_derivative;
		}
		
	}

	void INeuron::DeleteMemory()
	{
		hidden_state = 0;
		cell_state = 0;
		first_hidden_gradient = 0;
		first_cell_gradient = 0;
	}
};

