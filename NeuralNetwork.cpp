// NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "Derivatives.h"
#include "DenseConnections.h"

int main()
{
    size_t shape_length = 2;
    int *networkShape = new int[shape_length];
    networkShape[0] = 1;
    networkShape[1] = 1;

    size_t layer_i = 1;
    size_t neuron_i = 0;
    DenseConnections* connections = new DenseConnections(layer_i, neuron_i, networkShape);

    double* X = new double[1];
    X[0] = 3;

    double *Y = new double[1];
    Y[0] = 5.555;

    for (size_t i = 0; i < 25; i++)
    {
        double** activations = new double*[shape_length];
        double** costs = new double* [shape_length];
        activations[0] = X;
        for (size_t i = 0; i < shape_length; i++)
        {
            costs[i] = new double[networkShape[i]];
            if (i == 0)
                continue;
            activations[i] = new double[networkShape[i]];
        }
        double linear_function = connections->LinearFunction(activations);
        activations[1][0] = linear_function;
        double cost_derivative = Derivatives::SquaredMeanDerivative(linear_function, Y[0]);
        costs[1][0] = cost_derivative;
        
        double* gradients = new double[connections->GetWeightCount()];
        connections->GetGradients(0, gradients, activations, costs, cost_derivative);
        connections->SubtractGradients(gradients, 0, 0.01);
        std::cout << linear_function << "\n";
    }
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
