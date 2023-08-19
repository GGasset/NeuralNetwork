// NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <chrono>

#include "DenseNeuron.h"
#include "DenseLSTM.h"
#include "ActivationFunctions.h"
#include "Cost.h"
#include "NN.h"
#include "ValueGeneration.h"

int main()
{
	srand(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());

	size_t t_count = 1;
	double* X = new double[t_count];
	double* Y = new double[t_count];

	for (size_t t = 0; t < t_count; t++)
	{
		X[t] = 10;
		Y[t] = .2 + .1 * t;
	}

	size_t shape_length = 2;
	size_t* shape = new size_t[shape_length];
	shape[0] = 1;
	shape[1] = 1;
	/*shape[1] = 4;
	shape[2] = 2;
	shape[3] = 1;*/
	
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
			neurons[neuron_i] = new DenseLSTM(neuron_i + shape[0], previous_layer_start, prev_layer_length);
			neuron_i++;
		}

		previous_layer_start += shape[i - 1];
	}

	bool continue_training = true;
	NN* n = new NN(neurons, neuron_count, shape[0], shape[shape_length - 1]);
	double* output = 0;
	for (size_t i = 0; i < 10000; i++)
	{
		double* last_output = output;
		output = n->Execute(X, t_count);
		bool is_same_output = true;
		for (size_t j = 0; j < t_count; j++)
		{
			std::cout << "Y: " << output[j] << " | Y hat: " << Y[j] << " | i: " << i << std::endl;
			if (last_output == 0)
			{
				is_same_output = false;
			}
			else
			{
				double max_output = output[j] * (output[j] > last_output[j]) + last_output[j] * (last_output[j] > output[j]);
				double min_output = output[j] * (output[j] < last_output[j]) + last_output[j] * (last_output[j] < output[j]);
				is_same_output = is_same_output && ((max_output - min_output) < 10E-5);
			}
		}

		if (is_same_output)
		{
			continue_training = !is_same_output;
		}

		double learning_rate = 2;
		n->Supervised_batch(X, Y, learning_rate, t_count, Cost::SquaredMean);


		delete[] last_output;
	}
	n->free();
	delete n;

	delete[] output;
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
