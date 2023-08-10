// NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include "ActivationFunctions.h"
#include "Cost.h"
#include "NeuronLibrary.h"
#include "NN.h"
#include "ValueGeneration.h"

int main()
{
	size_t t_count = 2;
	double* X = new double[t_count];
	double* Y = new double[t_count];

	for (size_t t = 0; t < t_count; t++)
	{
		X[t] = 10;
		Y[t] = 0.5;
	}

	size_t shape_length = 6;
	size_t* shape = new size_t[shape_length];
	shape[0] = 1;
	shape[1] = 27;
	shape[2] = 12;
	shape[3] = 8;
	shape[4] = 4;
	shape[5] = 1;
	
	size_t neuron_count = 0;
	for (size_t i = 1; i < shape_length; i++)
	{
		neuron_count += shape[i];
	}

	ActivationFunctions::ActivationFunction activation_function = ActivationFunctions::ActivationFunction::Sigmoid;
	INeuron** neurons = new INeuron * [neuron_count];

	size_t neuron_i = 0;
	size_t previous_layer_start = 0;
	// TODO: do a switch case here using neuron_type and modularize main into a single module, saving X_length and that stuff
	for (size_t i = 1; i < shape_length; i++)
	{
		size_t prev_layer_length = shape[i - 1];
		for (size_t j = 0; j < shape[i] && (neuron_i < neuron_count); j++)
		{
			// Bug in previous_layer_length
			neurons[neuron_i] = new DenseNeuron(neuron_i + shape[0], previous_layer_start, prev_layer_length, activation_function);
			neuron_i++;
		}

		previous_layer_start += shape[i - 1];
	}

	NN* n = new NN(neurons, neuron_count, shape[0], shape[shape_length - 1]);
	for (size_t i = 0; i < 3000; i++)
	{
		double* output = n->Execute(X, t_count);
		for (size_t j = 0; j < t_count; j++)
		{
			std::cout << "Y: " << output[j] << " | Y hat: " << std::to_string(Y[j]) << " | i: " << i << "\n";
		}

		double learning_rate = 0.1;
		n->Supervised_batch(X, Y, learning_rate, t_count, Cost::SquaredMean);
	}
	n->free();
	delete n;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
