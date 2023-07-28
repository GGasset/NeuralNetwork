#include "IConnections.h"

#pragma once
class INeuron
{
protected:
	IConnections* connections;

public:
	virtual double* ExecuteStore(double**) = 0;

	virtual double Execute(double**) = 0;

	virtual double GetOutput(double*) = 0;

protected:
	virtual double* GetGradients(double neuron_cost, double** networkCosts, double** network_activations) = 0;

public:
	virtual double** GetGradients(double*** network_costs, double*** network_activations) = 0;

	virtual void SubtractGradients(double* neuronGradients) = 0;

	virtual void DeleteMemory() = 0;
};

