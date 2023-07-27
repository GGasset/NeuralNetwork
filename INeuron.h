#pragma once
class INeuron
{
	/// <summary>
	/// Input layer not included as its activations are the input
	/// </summary>
	int layer_i;

	virtual double* Execute();

	virtual double GetOutput(double*);
};

