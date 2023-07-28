#include "IConnections.h"

#pragma once
class INeuron
{
protected:
	IConnections* connections;
	double bias;

public:
	virtual double* ExecuteStore(double** networkActivations) = 0;

	virtual double Execute(double** network_activations) = 0;

	virtual double GetOutput(double* execute_store_output) = 0;

public:
	virtual double* GetGradients(double* execution_results, double neuron_cost, double** network_costs, double** network_activations) = 0;

public:
	/// <param name="network_gradients_over_t">
	///		Fourth dimension: t.
	///		Third-second dimension layer-neuron.
	///		First dimension: gradients calculated at GetGradients.
	/// </param>
	/// <param name="execution_results">
	///		Fourth dimension: t.
	///		Third-second dimension layer-neuron.
	///		First dimension: values calculated at ExecuteStore().
	/// </param>
	virtual void GetGradients(double**** execution_results, size_t calculated_steps, double**** output, double*** network_costs, double*** network_activations) = 0;

	virtual void SubtractGradients(double* neuron_gradients, double learning_rate) = 0;

	virtual void SubtractGradients(double**** network_gradients_over_t, size_t calculated_steps, double learning_rate) = 0;

	virtual void DeleteMemory() = 0;

	virtual void Free() = 0;
};

