#include "IConnections.h"

#pragma once
class INeuron
{
protected:
	IConnections connections;

public:
	virtual double* ExecuteStore(double**);

	virtual double Execute(double**);

	virtual double GetOutput(double*);

	virtual double* GetGradients(double neuron_cost, double** networkCosts, double** network_activations, double* neuronActivations);

	virtual void SubtractGradients(double* neuronGradients);

	virtual void DeleteMemory();
};

