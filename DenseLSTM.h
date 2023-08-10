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
		// execution_results[0] = output cell state
		// execution_results[1] = output hidden_state (neuron output)
		// execution_results[2] = linear_hidden_addition
		// execution_results[3] = linear_hidden_sigmoid output
		// execution_results[4] = forget weight multiplication
		// execution_results[5] = store sigmoid weight multiplication
		// execution_results[6] = linear_hidden_tanh output
		// execution_results[7] = store tanh weight multiplication
		// execution_results[8] = output_weight multiplication		
		// execution_results[9] = output cell state tanh	


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
			
			// Linear_hidden activations derivatives
			double linear_hidden_sigmoid_derivative = Derivatives::SigmoidDerivative(execution_results[current_execution_result_start + 2]);
			double linear_hidden_tanh_derivative = Derivatives::TanhDerivative(execution_results[current_execution_result_start + 2]);

			// Forget gate derivatives
			double forget_weight_multiplication_derivative = execution_results[current_execution_result_start + 3] + linear_hidden_sigmoid_derivative;
			double cell_state_multiplication_derivative = execution_results[current_execution_result_start + 4] * forget_weight_multiplication_derivative + execution_results[current_execution_result_start] * prev_cell_gradient;

			// Store gate derivatives
			double store_sigmoid_weight_derivative = execution_results[current_execution_result_start + 3] * linear_hidden_sigmoid_derivative;
			double store_tanh_weight_derivative = execution_results[current_execution_result_start + 6] * linear_hidden_tanh_derivative;

			double store_gate_multiplication_derivative =
				store_sigmoid_weight_derivative * execution_results[current_execution_result_start + 5]
				+
				store_tanh_weight_derivative * execution_results[current_execution_result_start + 7];

			double cell_state_addition_derivative = cell_state_multiplication_derivative + store_gate_multiplication_derivative;

			double cell_state_tanh_derivative = Derivatives::TanhDerivative(execution_results[current_execution_result_start + 1]);

			// Output gate
			double output_weight_multiplication_derivative = linear_hidden_sigmoid_derivative * execution_results[current_execution_result_start + 3];
			double output_gate_derivative =
				output_weight_multiplication_derivative * execution_results[current_execution_result_start + 8]
				+
				cell_state_tanh_derivative * execution_results[current_execution_result_start + 9];

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

