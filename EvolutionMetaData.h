#pragma once
class EvolutionMetaData
{
public:
	EvolutionMetaData()
	{

	}

	EvolutionMetaData(EvolutionMetaData& to_copy)
	{
		mutation_chance_of_new_neuron_chance_mutation_probability = to_copy.mutation_chance_of_new_neuron_chance_mutation_probability;
		max_mutation_of_new_neuron_chance_mutation_probability = to_copy.max_mutation_of_new_neuron_chance_mutation_probability;

		mutation_chance_of_new_neuron_chance_max_mutation = to_copy.mutation_chance_of_new_neuron_chance_max_mutation;
		max_mutation_of_new_neuron_chance_max_mutation = to_copy.max_mutation_of_new_neuron_chance_max_mutation;

		new_neuron_chance_mutation_probability = to_copy.new_neuron_chance_mutation_probability;
		new_neuron_chance_max_mutation = to_copy.new_neuron_chance_max_mutation;

		new_neuron_chance = to_copy.new_neuron_chance;

		mutation_chance_of_new_neuron_in_new_layer_chance_mutation_probability = to_copy.mutation_chance_of_new_neuron_in_new_layer_chance_mutation_probability;
		max_mutation_of_new_neuron_in_new_layer_chance_mutation_probability = to_copy.max_mutation_of_new_neuron_in_new_layer_chance_mutation_probability;

		mutation_chance_of_new_neuron_in_new_layer_chance_max_mutation = to_copy.mutation_chance_of_new_neuron_in_new_layer_chance_max_mutation;
		max_mutation_of_new_neuron_in_new_layer_chance_max_mutation = to_copy.max_mutation_of_new_neuron_in_new_layer_chance_max_mutation;

		new_neuron_in_new_layer_chance_mutation_probability = to_copy.new_neuron_in_new_layer_chance_mutation_probability;
		new_neuron_in_new_layer_chance_max_mutation = to_copy.new_neuron_in_new_layer_chance_max_mutation;

		new_neuron_in_new_layer_chance = to_copy.new_neuron_in_new_layer_chance;

		mutation_chance_of_weight_mutation_probability_mutation_chance = to_copy.mutation_chance_of_weight_mutation_probability_mutation_chance;
		max_mutation_of_weight_mutation_probability_mutation_chance = to_copy.max_mutation_of_weight_mutation_probability_mutation_chance;

		mutation_chance_of_weight_mutation_probability_max_mutation = to_copy.mutation_chance_of_weight_mutation_probability_max_mutation;
		max_mutation_of_weight_mutation_probability_max_mutation = to_copy.max_mutation_of_weight_mutation_probability_max_mutation;

		weight_mutation_probability_mutation_chance = to_copy.weight_mutation_probability_mutation_chance;
		weight_mutation_probabily_max_mutation = to_copy.weight_mutation_probabily_max_mutation;

		weight_mutation_probability = to_copy.weight_mutation_probability;

		mutation_chance_of_max_weight_mutation_mutation_chance = mutation_chance_of_max_weight_mutation_mutation_chance;
		max_mutation_of_max_weight_mutation_mutation_chance = max_mutation_of_max_weight_mutation_mutation_chance;

		mutation_chance_of_max_weight_mutation_max_mutation = mutation_chance_of_max_weight_mutation_max_mutation;
		max_mutation_of_max_weight_mutation_max_mutation = max_mutation_of_max_weight_mutation_max_mutation;


		max_weight_mutation_mutation_chance = max_weight_mutation_mutation_chance;
		max_weight_mutation_max_mutation = max_weight_mutation_max_mutation;


		max_weight_mutation = max_weight_mutation;

	}

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

