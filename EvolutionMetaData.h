#pragma once
class EvolutionMetaData
{
public:
	// Neuron addition 

	double mutation_chance_of_new_neuron_chance_mutation_probability = 0;
	double max_mutation_of_new_neuron_chance_mutation_probability = 0;

	double mutation_chance_of_new_neuron_chance_max_mutation = 0;
	double max_mutation_of_new_neuron_chance_max_mutation = 0;


	double new_neuron_chance_mutation_probability = 0;
	double new_neuron_chance_max_mutation = 0;


	double new_neuron_chance = 0;



	double mutation_chance_of_new_neuron_in_new_layer_chance_mutation_probability = 0;
	double max_mutation_of_new_neuron_in_new_layer_chance_mutation_probability = 0;

	double mutation_chance_of_new_neuron_in_new_layer_chance_max_mutation = 0;
	double max_mutation_of_new_neuron_in_new_layer_chance_max_mutation = 0;
	
	
	double new_neuron_in_new_layer_chance_mutation_probability = 0;
	double new_neuron_in_new_layer_chance_max_mutation = 0;

	double new_neuron_in_new_layer_chance = 0;


	// Weight mutation
		// * Weight mutation chance

	double mutation_chance_of_weight_mutation_probability_mutation_chance = 0;
	double max_mutation_of_weight_mutation_probability_mutation_chance = 0;

	double mutation_chance_of_weight_mutation_probability_max_mutation = 0;
	double max_mutation_of_weight_mutation_probability_max_mutation = 0;


	double weight_mutation_probability_mutation_chance = 0;
	double weight_mutation_probabily_max_mutation = 0;


	double weight_mutation_probability = 0;



		// * Max weight mutation

	double mutation_chance_of_max_weight_mutation_mutation_chance = 0;
	double max_mutation_of_max_weight_mutation_mutation_chance = 0;

	double mutation_chance_of_max_weight_mutation_max_mutation = 0;
	double max_mutation_of_max_weight_mutation_max_mutation = 0;


	double max_weight_mutation_mutation_chance = 0;
	double max_weight_mutation_max_mutation = 0;


	double max_weight_mutation = 0;
};

