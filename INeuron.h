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

	virtual double GetOutput(double* network_activations) = 0;

protected:
	virtual double* GetGradients(double neuron_cost, double** networkCosts, double** network_activations) = 0;

public:
	/// <param name="network_gradients_over_t">
	///		Fourth dimension: t.
	///		Third-second dimension layer-neuron.
	///		First dimension: gradients calculated at GetGradients.
	/// </param>
	virtual void GetGradients(size_t calculated_steps, double**** output, double*** network_costs, double*** network_activations) = 0;

	virtual void SubtractGradients(double* neuronGradients) = 0;

	virtual void DeleteMemory() = 0;
};

