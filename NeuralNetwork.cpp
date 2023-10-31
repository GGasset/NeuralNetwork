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
	NN::SetValueGenerationSeed();
	// Improve multithreading to assignate more work to each thread
	bool use_multithreading = false;

	size_t t_count = 2;
	double* X = new double[t_count];
	double* Y = new double[t_count];

	for (size_t t = 0; t < t_count; t++)
	{
		X[t] = 2;
		Y[t] = .5 + .3 * t;
	}

	size_t shape_length = 4;
	size_t* shape = new size_t[shape_length];
	shape[0] = 1;
	shape[1] = 20;
	shape[2] = 10;
	shape[3] = 1;
	
	size_t neuron_count = 0;
	for (size_t i = 1; i < shape_length; i++)
	{
		neuron_count += shape[i];
	}

	ActivationFunctions::ActivationFunction activation_function = ActivationFunctions::ActivationFunction::Sigmoid;
	INeuron** neurons = new INeuron * [neuron_count];
	NN::NeuronTypeIdentifier* neurons_id = new NN::NeuronTypeIdentifier[neuron_count];

	int8_t weight_direction = 1;
	size_t neuron_i = 0;
	size_t previous_layer_start = 0;
	// TODO: do a switch case here using neuron_type and modularize main into a single module, saving X_length and that stuff
	for (size_t i = 1; i < shape_length; i++)
	{
		size_t prev_layer_length = shape[i - 1];
		for (size_t j = 0; j < shape[i] && (neuron_i < neuron_count); j++)
		{
			neurons[neuron_i] = new DenseLSTM(neuron_i + shape[0], previous_layer_start, prev_layer_length, weight_direction);
			neurons_id[neuron_i] = NN::DenseLSTMId;
			neuron_i++;
		}

		previous_layer_start += shape[i - 1];
	}

	NN* n = new NN(neurons, neuron_count, shape[0], shape[shape_length - 1], shape, shape_length, neurons_id);
	std::cout << "Before:" << std::endl;
	double* before_output = n->Execute(X, t_count, true, use_multithreading);
	for (size_t i = 0; i < t_count; i++)
	{
		std::cout << before_output[i] << std::endl;
	}
	delete[] before_output;

	//std::string path = "D:\\NN\\tests\\01";
	std::string path = "C:\\Users\\GG\\Documents\\NNs";
	n->Save(path);
	n->free();
	n = NN::Load(path);

	std::cout << std::endl << "After being loaded from disk:" << std::endl;
	double* after_output = n->Execute(X, t_count, true, use_multithreading);
	for (size_t i = 0; i < t_count; i++)
	{
		std::cout << after_output[i] << std::endl;
	}
	delete[] after_output;
	std::cin.get();
	std::cout << std::endl;

	double* previous_cost = 0;
	double* output = 0;
	bool continue_training = true;
	double learning_rate = 1;
	for (size_t i = 0; i < 1000/* && continue_training*/; i++)
	{
		double* last_output = output;
		output = n->Execute(X, t_count, true, use_multithreading);
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
				is_same_output = is_same_output && ((max_output - min_output) < 10E-15);
			}
		}

		continue_training = !is_same_output;

		double cost = n->SupervisedBatch(X, Y, t_count, Cost::SquaredMean, &learning_rate, NN::LearningEffectiveness, false, previous_cost, true, .2);
		previous_cost = &cost;

		delete[] last_output;
	}
	n->free();
	delete n;

	delete[] output;
	delete[] X;
	delete[] Y;
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
