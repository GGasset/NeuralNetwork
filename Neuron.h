#include "INeuron.h"
# include "ActivationFunctions.h"
#include "Derivatives.h"

#pragma once
class Neuron : public INeuron
{
public:
	ActivationFunctions::ActivationFunction activation_function;


	void Free()
	{
		connections->Free();
	}
};

