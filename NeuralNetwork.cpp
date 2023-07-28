// NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "Derivatives.h"
#include "DenseConnections.h"
#include "Neuron.h"
//#include "ValueGeneration.h"

int main()
{
    // Singleton prototype based on linear function
    /*size_t shape_length = 2;
    size_t *networkShape = new size_t[shape_length];
    networkShape[0] = 1;
    networkShape[1] = 1;

    size_t layer_i = 1;
    size_t neuron_i = 0;
    DenseConnections* connections = new DenseConnections(layer_i, neuron_i, networkShape);

    double* X = new double[1];
    X[0] = 3;

    double *Y = new double[1];
    Y[0] = 5.555;

    for (size_t j = 0; j < 25; j++)
    {
        double** activations = new double*[shape_length];
        double** costs = new double* [shape_length];
        activations[0] = X;
        for (size_t j = 0; j < shape_length; j++)
        {
            costs[j] = new double[networkShape[j]];
            if (j == 0)
                continue;
            activations[j] = new double[networkShape[j]];
        }
        double linear_function = connections->LinearFunction(activations);
        activations[1][0] = linear_function;
        double cost_derivative = Derivatives::SquaredMeanDerivative(linear_function, Y[0]);
        costs[1][0] = cost_derivative;
        
        double* gradients = new double[connections->GetWeightCount()];
        connections->GetGradients(0, gradients, activations, costs, cost_derivative);
        connections->SubtractGradients(gradients, 0, 0.01);
        std::cout << linear_function << "\n";
    }*/

    // Singleton
    size_t shape_length = 2;
    size_t* network_shape = new size_t[shape_length];
    network_shape[0] = 3;
    network_shape[1] = 1;

    Neuron* singleton = new Neuron((IConnections*)(new DenseConnections(1, 0, network_shape)), 1, ActivationFunctions::Sigmoid);
    double* X1 = new double[3];
    X1[0] = 3;
    X1[1] = -5;
    X1[2] = 4;

    double* X2 = new double[3];
    X2[0] = -5;
    X2[1] = 3;
    X2[2] = -7;

    double* Y1 = new double(0.6);
    double* Y2 = new double(0.4);

    for (size_t i = 0; i < 50; i++)
    {
        double** network_activations = new double*[shape_length];
        bool Y_option = ValueGeneration::NextDouble() >= 0.5;
        network_activations[0] = Y_option ? X1 : X2;
        network_activations[1] = new double[network_shape[1]];

        double* execution_results = singleton->ExecuteStore(network_activations);
        double output = singleton->GetOutput(execution_results);
        std::cout << i << " | " << Y_option << " | " << output << "\n";

        network_activations[0] = Y_option ? X1 : X2;

        double output_cost = Derivatives::SquaredMeanDerivative(output, Y_option ? *Y1 : *Y2);
        double** network_costs = new double* [shape_length];
        network_costs[0] = new double[network_shape[0]];
        network_costs[1] = new double[1];
        for (size_t j = 0; j < network_shape[0]; j++)
        {
            network_costs[0][j] = 0.0;
        }
        network_costs[1][0] = output_cost;

        double* gradients = singleton->GetGradients(execution_results, output_cost, network_costs, network_activations);
        singleton->SubtractGradients(gradients, 0.15);


        delete[] execution_results;
        delete[] gradients;
        for (size_t j = 0; j < shape_length; j++)
        {
            delete[] network_costs[j];
            if (j == 0)
                continue;
            delete[] network_activations[j];
        }
        delete[] network_costs;
        delete[] network_activations;
    }

    delete[] network_shape;
    singleton->Free();

    delete[] X1;
    delete[] X2;

    delete Y1;
    delete Y2;
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
