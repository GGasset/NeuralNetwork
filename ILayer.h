#include "INeuron.h"

#pragma once
/// <summary>
/// The purpose of a layer is a easy instantiation of layer and connection, not functions.
/// </summary>
class ILayer
{
private:
	size_t layer_length = -1;
public:
	INeuron* neurons = 0;

	void free()
	{
		for (size_t i = 0; i < layer_length; i++)
		{
			neurons[i].Free();
			delete (neurons + i);
		}
		delete[] neurons;
	}
};

