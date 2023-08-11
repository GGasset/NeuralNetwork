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

	double first_hidden_derivative = 0;
	double first_cell_derivative = 0;

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

		neuron_written_gradient_count = 5;
		neuron_written_execution_results_count = 10;

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
		// Derivarives positions
		// Relative indexing*
		// Derivatives[0] = output cell state derivative (cell state addition derivative);
		// Derivatives[1] = output hidden state derivative (output gate multiplication derivative);
		// Derivatives[2] = linear hidden sigmoid derivative;
		// Derivatives[3] = linear hidden tanh derivative;
		// Derivatives[4] = forget gate weight multiplication derivative;
		// Derivatives[5] = cell state forget multiplication derivative;
		// Derivatives[6] = store gate sigmoid weight multiplication derivative;
		// Derivatives[7] = store gate tanh weight multiplication derivative;
		// Derivatives[8] = store gate multiplication derivative;
		// Derivatives[9] = output gate weight multiplication derivative;
		// Derivatives[10] = output cell state tanh derivative;

		// Gradient positions
		// Relative indexing*
		// gradients[0] = output_hiddden_state gradient
		// gradients[1] = output_cell_state_gradient
		// gradients[2] = 
		// gradients[3] = 
		// gradients[4] = 
		// gradients[5] = 
		// gradients[6] = 
		// gradients[7] = 

		size_t derivative_per_t_count = 11;
		double* derivatives = new double[derivative_per_t_count * t_count];
		for (size_t t = 0; t < t_count; t++)
		{
			size_t current_execution_result_start = connections->network_execution_results_value_count * t_count + self_execution_results_start_i;
			size_t previous_derivatives_start = derivative_per_t_count * (t - 1);

			double prev_cell_state_derivative = t == 0 ? first_cell_derivative : derivatives[previous_derivatives_start];
			double prev_hidden_state_derivative = t == 0 ? first_hidden_derivative : derivatives[previous_derivatives_start + 1];
			
			// Derivatives
			size_t current_derivatives_start = derivative_per_t_count * t;

			// Linear_hidden activations derivatives
			double linear_hidden_sigmoid_derivative = Derivatives::SigmoidDerivative(execution_results[current_execution_result_start + 2]);
			double linear_hidden_tanh_derivative = Derivatives::TanhDerivative(execution_results[current_execution_result_start + 2]);

			derivatives[current_derivatives_start + 2] = linear_hidden_sigmoid_derivative;
			derivatives[current_derivatives_start + 3] = linear_hidden_tanh_derivative;

			// Forget gate derivatives
			double forget_weight_multiplication_derivative = execution_results[current_execution_result_start + 3] + linear_hidden_sigmoid_derivative;
			derivatives[current_derivatives_start + 4] = forget_weight_multiplication_derivative;

			double cell_state_multiplication_derivative = 
				execution_results[current_execution_result_start + 4] * forget_weight_multiplication_derivative
				+
				execution_results[current_execution_result_start] * prev_cell_state_derivative;
			derivatives[current_derivatives_start + 5] = cell_state_multiplication_derivative;

			// Store gate derivatives
			double store_sigmoid_weight_derivative = execution_results[current_execution_result_start + 3] * linear_hidden_sigmoid_derivative;
			double store_tanh_weight_derivative = execution_results[current_execution_result_start + 6] * linear_hidden_tanh_derivative;

			derivatives[current_derivatives_start + 6] = store_sigmoid_weight_derivative;
			derivatives[current_derivatives_start + 7] = store_tanh_weight_derivative;

			double store_gate_multiplication_derivative =
				store_sigmoid_weight_derivative * execution_results[current_execution_result_start + 5]
				+
				store_tanh_weight_derivative * execution_results[current_execution_result_start + 7];

			derivatives[current_derivatives_start + 8] = store_gate_multiplication_derivative;

			double cell_state_addition_derivative = cell_state_multiplication_derivative + store_gate_multiplication_derivative;
			derivatives[current_derivatives_start] = cell_state_addition_derivative;

			double cell_state_tanh_derivative = Derivatives::TanhDerivative(execution_results[current_execution_result_start + 1]);
			derivatives[current_derivatives_start + 10] = cell_state_tanh_derivative;

			// Output gate
			double output_weight_multiplication_derivative = linear_hidden_sigmoid_derivative * execution_results[current_execution_result_start + 3];
			derivatives[current_derivatives_start + 9] = output_weight_multiplication_derivative;

			double output_gate_derivative =
				output_weight_multiplication_derivative * execution_results[current_execution_result_start + 8]
				+
				cell_state_tanh_derivative * execution_results[current_execution_result_start + 9];
			derivatives[current_derivatives_start + 1] = output_gate_derivative;
		}
		
		for (int t = t_count - 1; t >= 0; t--)
		{

		}
	}

	void INeuron::DeleteMemory()
	{
		hidden_state = 0;
		cell_state = 0;
		first_hidden_derivative = 0;
		first_cell_derivative = 0;
	}
};

