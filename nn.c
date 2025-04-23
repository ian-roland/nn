/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include "nn.h"

// Private functions

typedef float (*activation_function_t)(float a, bool derivative);

// Null activation function
static float activation_function_none(float a, bool derivative)
{
	return 0;
}

// Identity activation function
static float activation_function_identity(float a, bool derivative)
{
	if (derivative)
		return 1;
	return a;
}

// Linear activation function
static float activation_function_linear(float a, bool derivative)
{
	if (derivative)
		return 1;
	return a;
}

// Rectified Linear Unit (ReLU) activation function
static float activation_function_relu(float a, bool derivative)
{
	if (a >= 0)
		return (derivative ? 1 : a);
	return 0;
}

// Leaky Rectified Linear Unit (Leaky ReLU) activation function
static float activation_function_leaky_relu(float a, bool derivative)
{
	if (a > 0)
		return (derivative ? 1 : a);
	return (derivative ? 0.01 : a * 0.01f);
}

// Exponential Linear Unit (ELU) activation function
static float activation_function_elu(float a, bool derivative)
{
	if (a >= 0)
		return (derivative ? 1 : a);
	return (derivative ? activation_function_elu(a, false) : expf(a) - 1);
}

// Threshold activation function
static float activation_function_threshold(float a, bool derivative)
{
	if (derivative)
		return 0;
	return a > 0;
}

// Sigmoid activation function (aka Logistic, aka Soft Step)
static float activation_function_sigmoid(float a, bool derivative)
{
	if (derivative) {
		float f = activation_function_sigmoid(a, false);
		return(f * (1.0f - f));
	}
	return 1.0f / (1.0f + expf(-a));
}

// Sigmoid activation function using a lookup table
static float activation_function_sigmoid_fast(float a, bool derivative)
{
	// Sigmoid outputs
	const float s[] = {0.0,0.000045,0.000123,0.000335,0.000911,0.002473,0.006693,0.017986,0.047426,0.119203,0.268941,0.500000,0.731059,0.880797,0.952574,0.982014,0.993307,0.997527,0.999089,0.999665,0.999877,0.999955,1.0};
	// Derivative of the sigmoid
	const float ds[] = {0.0,0.000045,0.000123,0.000335,0.000910,0.002467,0.006648,0.017663,0.045177,0.104994,0.196612,0.250000,0.196612,0.104994,0.045177,0.017663,0.006648,0.002466,0.000910,0.000335,0.000123,0.000045,0.0};
	int index;
	float fraction = 0;

	index = floor(a) + 11;
	if (index < 0)
		index = 0;
	else if (index > 21)
		index = 21;
	else
		fraction = a - floor(a);
	if (derivative)
		return ds[index] + (ds[index + 1] - ds[index]) * fraction;
	return s[index] + (s[index + 1] - s[index]) * fraction;
}

// Tanh activation function
static float activation_function_tanh(float a, bool derivative)
{
	if (derivative)
		return 1.0 - activation_function_tanh(a, false) * activation_function_tanh(a, false);
	return (2.0 / (1.0 + expf(-2.0 * a))) - 1.0;
}

// Fast Tanh activation function
static float activation_function_tanh_fast(float a, bool derivative)
{
	if (derivative)
		return 1.0f / ((1.0f + abs(a)) * (1.0f + abs(a)));
	return a / (1.0f + abs(a));
}

// These must be in the same order as the enum activation_function_type
static activation_function_t activation_function[] = {
	activation_function_none,
	activation_function_identity,
	activation_function_linear,
	activation_function_relu,
	activation_function_leaky_relu,
	activation_function_elu,
	activation_function_threshold,
	activation_function_sigmoid,
	activation_function_sigmoid_fast,
	activation_function_tanh,
	activation_function_tanh_fast
};

// Computes the error given a cost function
static float error(float a, float b)
{
	return 0.5f * (a - b) * (a - b);
}

// Computes derivative of the error through the derivative of the cost function
static float error_derivative(float a, float b)
{
	return a - b;
}

static void forward_propagation(nn_t *nn)
{
	float sum;
	int i, j, k;

	// Calculate neuron values in each layer
	for (i = 1; i < nn->depth; i++) {
		for (j = 0; j < nn->width[i]; j++) {
			sum = 0;
			for (k = 0; k < nn->width[i - 1]; k++)
				sum += nn->neuron[i - 1][k] * nn->weight[i][j][k];
			sum += nn->bias[i];
			nn->neuron[i][j] = activation_function[nn->activation[i]](sum, false);
			// Store the preactivation value of this neuron for later use in backpropagation
			nn->preact[i][j] = sum;
		}
	}
}

// Public functions

nn_t *nn_init(void)
{
	nn_t *nn;

	nn = (nn_t *)malloc(sizeof(nn_t));
	if (NULL == nn)
		return NULL;
	nn->depth = 0;
	nn->width = NULL;
	nn->weight = NULL;
	nn->weight_adj = NULL;
	nn->neuron = NULL;
	nn->loss = NULL;
	nn->preact = NULL;
	nn->bias = NULL;
	nn->activation = NULL;
	nn->quantized_network = NULL;
	return nn;
}

void nn_free(nn_t *nn) {
    if (!nn) return;

    // Free quantized data
    if (nn->quantized_network) {
        nn_quantized_t* q = nn->quantized_network;
        
        // Free quantized weights/scales/biases
        for (uint32_t layer = 1; layer < q->depth; layer++) {
            for (uint32_t neuron = 0; neuron < q->width[layer]; neuron++) {
                free(q->quantized_weights[layer][neuron]);
            }
            free(q->quantized_weights[layer]);
            free(q->weight_scales[layer]);
            free(q->quantized_biases[layer]);
        }
        free(q->quantized_weights);
        free(q->weight_scales);
        free(q->quantized_biases);
        free(q->bias_scales);
        free(q->width);
        free(q);
        nn->quantized_network = NULL;
    }

    // Free original network data (only if they exist)
    if (nn->weight) {
        for (int layer = 1; layer < nn->depth; layer++) {
            if (nn->weight[layer]) {
                for (int i = 0; i < nn->width[layer]; i++) {
                    free(nn->weight[layer][i]);
                }
                free(nn->weight[layer]);
            }
        }
        free(nn->weight);
    }

    if (nn->weight_adj) {
        // Similar to weight cleanup
    }

    if (nn->neuron) {
        for (int layer = 1; layer < nn->depth; layer++) {
            free(nn->neuron[layer]);
        }
        free(nn->neuron);
    }

    // Free other fields
    free(nn->loss);
    free(nn->preact);
    free(nn->width);
    free(nn->activation);
    free(nn->bias);
    free(nn);
}

int nn_add_layer(nn_t *nn, int width, int activation, float bias)
{
	nn->depth++;
	nn->width = (uint32_t *)realloc(nn->width, nn->depth * sizeof(*nn->width));
	if (NULL == nn->width)
		return 1;
	nn->width[nn->depth - 1] = width;
	nn->activation = (uint8_t *)realloc(nn->activation, nn->depth * sizeof(*nn->activation));
	if (NULL == nn->activation)
		return 1;
	nn->activation[nn->depth - 1] = activation;
	nn->bias = (float *)realloc(nn->bias, nn->depth * sizeof(*nn->bias));
	if (NULL == nn->bias)
		return 1;
	nn->bias[nn->depth - 1] = bias;
	nn->neuron = (float **)realloc(nn->neuron, nn->depth * sizeof(float *));
	if (NULL == nn->neuron)
		return 1;
	nn->loss = (float **)realloc(nn->loss, nn->depth * sizeof(float *));
	if (NULL == nn->loss)
		return 1;
	nn->preact = (float **)realloc(nn->preact, nn->depth * sizeof(float *));
	if (NULL == nn->preact)
		return 1;
	if (nn->depth > 1) {
		nn->neuron[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
		if (NULL == nn->neuron[nn->depth - 1])
			return 1;
		nn->loss[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
		if (NULL == nn->loss[nn->depth - 1])
			return 1;
		nn->preact[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
		if (NULL == nn->preact[nn->depth - 1])
			return 1;
	}
	nn->weight = (float ***)realloc(nn->weight, (nn->depth) * sizeof(float **));
	if (NULL == nn->weight)
		return 1;
	nn->weight_adj = (float ***)realloc(nn->weight_adj, (nn->depth) * sizeof(float **));
	if (NULL == nn->weight_adj)
		return 1;
	if (nn->depth > 1) {
		nn->weight[nn->depth - 1] = (float **)malloc((nn->width[nn->depth - 1]) * sizeof(float *));
		if (NULL == nn->weight[nn->depth - 1])
			return 1;
		nn->weight_adj[nn->depth - 1] = (float **)malloc((nn->width[nn->depth - 1]) * sizeof(float *));
		if (NULL == nn->weight_adj[nn->depth - 1])
			return 1;
		for (int neuron = 0; neuron < nn->width[nn->depth - 1]; neuron++) {
			nn->weight[nn->depth - 1][neuron] = (float *)malloc((nn->width[nn->depth - 2]) * sizeof(float));
			if (NULL == nn->weight[nn->depth - 1][neuron])
				return 1;
			nn->weight_adj[nn->depth - 1][neuron] = (float *)malloc((nn->width[nn->depth - 2]) * sizeof(float));
			if (NULL == nn->weight_adj[nn->depth - 1][neuron])
				return 1;
			// Randomize the weights in this layer using uniform Xavier initialization
			// Range = +/- sqrt(6 / (#inputs + #outputs))
			for (int i = 0; i < nn->width[nn->depth - 2]; i++) {
				nn->weight[nn->depth - 1][neuron][i] = sqrtf(6.0f / (nn->width[nn->depth - 1] + nn->width[nn->depth - 2])) * 2.0f * (rand() / (float)RAND_MAX - 0.5f);
			}
		}
	}
	return 0;
}

// Trains a nn with a given input and target output at a specified learning rate.
// The rate (or step size) controls how far in the search space to move against the
// gradient in each iteration of the algorithm.
// Returns the total error between the target and the output of the neural network.
float nn_train(nn_t *nn, float *inputs, float *targets, float rate)
{
	float sum;
	int i, j, k;
	float err = 0;

	nn->neuron[0] = inputs;
	forward_propagation(nn);
	// Perform back propagation using gradient descent, which is an optimization algorithm that follows the
	// negative gradient of the objective function to find the minimum of the function.
	// Start at the output layer, and work backward toward the input layer, adjusting weights along the way.
	// Calculate the error aka loss aka delta at the output.
	// Select last layer (output layer)
	i = nn->depth - 1;
	for (j = 0; j < nn->width[i]; j++) {
		// Calculate the loss between the target and the outputs of the last layer
		nn->loss[i][j] = error_derivative(targets[j], nn->neuron[i][j]);
		err += error(targets[j], nn->neuron[i][j]);
	}
	// Calculate losses throughout the inner layers, not including layer 0 which can have no loss
	for (i = nn->depth - 2; i > 0 ; i--) {
		for (j = 0; j < nn->width[i]; j++) {
			sum = 0;
			for (k = 0; k < nn->width[i + 1]; k++)
				sum += nn->loss[i + 1][k] * activation_function[nn->activation[i]](nn->preact[i + 1][k], true) * nn->weight[i + 1][k][j];
			nn->loss[i][j] = sum;
		}
	}
	// Calculate the weight adjustments
	// The weights cannot be updated while back-propagating, because back propagating each layer depends on the next layer's weights.
	// So we save the weight adjustments in a temporary array and apply them all at once later.	
	for (i = nn->depth - 1; i > 0 ; i--)
		for (j = 0; j < nn->width[i]; j++)
			for (k = 0; k < nn->width[i - 1]; k++)
				nn->weight_adj[i][j][k] = nn->loss[i][j] * activation_function[nn->activation[i]](nn->preact[i][j], true) * nn->neuron[i - 1][k];
	// Apply the weight adjustments
	for (i = nn->depth - 1; i > 0 ; i--)
		for (j = 0; j < nn->width[i]; j++)
			for (k = 0; k < nn->width[i - 1]; k++)
				nn->weight[i][j][k] += nn->weight_adj[i][j][k] * rate;
	return err;
}

// Returns an output prediction given an input
float *nn_predict(nn_t *nn, float *inputs)
{
	nn->neuron[0] = inputs;
	forward_propagation(nn);
	// Return a pointer to the output layer
	return nn->neuron[nn->depth - 1];
}

float *nn_predict_quantized(nn_t *network, float *input) {
    if (!network || !network->quantized_network || !input)
        return NULL;

    nn_quantized_t *quantized = network->quantized_network;
    const uint32_t depth = network->depth;
    const uint32_t *width = network->width;

    float *activations = malloc(sizeof(float) * width[0]);
    if (!activations) return NULL;
    memcpy(activations, input, sizeof(float) * width[0]);

    for (uint32_t layer = 1; layer < depth; layer++) {
        const uint32_t curr_width = width[layer];
        float *new_activations = malloc(sizeof(float) * curr_width);
        if (!new_activations) {
            free(activations);
            return NULL;
        }

        for (uint32_t neuron = 0; neuron < curr_width; neuron++) {
            float sum = 0.0f;
            const uint32_t prev_width = width[layer-1];

            for (uint32_t w = 0; w < prev_width; w++) {
                const float weight = quantized->quantized_weights[layer][neuron][w] * quantized->weight_scales[layer][neuron];
                sum += weight * activations[w];
            }

            const float bias = quantized->quantized_biases[layer][neuron] * quantized->bias_scales[layer];
            sum += bias;

            new_activations[neuron] = activate(sum, network->activation[layer]);
        }

        free(activations);
        activations = new_activations;
    }

    return activations;
}

// Loads a neural net model file
nn_t *nn_load(char *path) {
    FILE *file;
    nn_t *nn;
    int width = 0;
    int activation = ACTIVATION_FUNCTION_TYPE_NONE;
    float bias = 0;
    int layer, i, j;
    int depth;
    int flag;

    file = fopen(path, "r");
    if (file == NULL) {
        return NULL;
    }

    nn = nn_init();
    if (nn == NULL) {
        fclose(file);
        return NULL;
    }

    // Read and check the flag
    if (fscanf(file, "%d\n", &flag) != 1) {
        fclose(file);
        nn_free(nn);
        return NULL;
    }

    // Read and check depth
    if (fscanf(file, "%d\n", &depth) != 1) {
        fclose(file);
        nn_free(nn);
        return NULL;
    }

    // Read layer configurations
    for (int i = 0; i < depth; i++) {
        if (fscanf(file, "%d %d %f\n", &width, &activation, &bias) != 3) {
            fclose(file);
            nn_free(nn);
            return NULL;
        }

        if (nn_add_layer(nn, width, activation, bias) != 0) {
            fclose(file);
            nn_free(nn);
            return NULL;
        }
    }

    // Read weights with error checking
    for (layer = 1; layer < nn->depth; layer++) {
        for (i = 0; i < nn->width[layer]; i++) {
            for (j = 0; j < nn->width[layer - 1]; j++) {
                if (fscanf(file, "%f\n", &nn->weight[layer][i][j]) != 1) {
                    fclose(file);
                    nn_free(nn);
                    return NULL;
                }
            }
        }
    }

    fclose(file);
    return nn;
}

// Saves a neural net model file
int nn_save(nn_t *nn, char *path)
{
	int layer, i, j;
	FILE *file;

	file = fopen(path, "w");
	if (NULL == file)
		return 1;
	// File format:
	// depth
	// width, activation, bias
	// weight

	fprintf(file, "0\n");
	fprintf(file, "%d\n", nn->depth);
	for (i = 0; i < nn->depth; i++)
		fprintf(file, "%d %d %f\n", nn->width[i], nn->activation[i], nn->bias[i]);
	for (layer = 1; layer < nn->depth; layer++)
		for (i = 0; i < nn->width[layer]; i++)
			for (j = 0; j < nn->width[layer - 1]; j++)
				fprintf(file, "%f\n", nn->weight[layer][i][j]);
	fclose(file);
	return 0;
}

void nn_free_quantized(nn_quantized_t* quantized_network) {
    if (!quantized_network) return;

    for (int layer = 1; layer < quantized_network->depth; layer++) { 
        int curr_width = quantized_network->width[layer]; 
        
        for (int neuron = 0; neuron < curr_width; neuron++) {
            free(quantized_network->quantized_weights[layer][neuron]); 
        }
        free(quantized_network->quantized_weights[layer]); 
        free(quantized_network->weight_scales[layer]); 
        free(quantized_network->quantized_biases[layer]); 
    }

    free(quantized_network->quantized_weights);
    free(quantized_network->weight_scales);
    free(quantized_network->quantized_biases);
    free(quantized_network->bias_scales);
    free(quantized_network->width); 
    free(quantized_network);
}

nn_t* nn_load_quantized(const char* path) {
    FILE* file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Failed to open quantized model file: %s\n", path);
        return NULL;
    }

    // Initialize the original network
    nn_t* network = malloc(sizeof(nn_t));
    if (!network) {
        fclose(file);
        return NULL;
    }

    network->quantized_network = NULL;
	network->weight = NULL;
    network->weight_adj = NULL;
    network->neuron = NULL;
    network->loss = NULL;
    network->preact = NULL;

    // Read the format flag (skip)
    int flag;
    if (fscanf(file, "%d\n", &flag) != 1) {
        fclose(file);
        free(network);
        return NULL;
    }

    // Read depth
    if (fscanf(file, "%u\n", &network->depth) != 1) {
        fprintf(stderr, "Error reading network depth\n");
        goto error;
    }

    // Allocate width/activation/bias arrays for the original network
    network->width = malloc(sizeof(uint32_t) * network->depth);
    network->activation = malloc(sizeof(uint8_t) * network->depth);
    network->bias = malloc(sizeof(float) * network->depth);

    // Read layer configurations
    for (uint32_t i = 0; i < network->depth; i++) {
        uint32_t width;
        uint8_t activation;
        float bias;
        if (fscanf(file, "%u %hhu %f\n", &width, &activation, &bias) != 3) {
            fprintf(stderr, "Error reading layer %u configuration\n", i);
            goto error;
        }
        network->width[i] = width;
        network->activation[i] = activation;
        network->bias[i] = bias;
    }

    // Allocate and load quantized data into nn_quantized_t
    nn_quantized_t* quantized = malloc(sizeof(nn_quantized_t));
    if (!quantized) goto error;
    network->quantized_network = quantized;

    // Initialize quantized data arrays
    quantized->depth = network->depth;
    quantized->width = malloc(sizeof(uint32_t) * quantized->depth);
    memcpy(quantized->width, network->width, sizeof(uint32_t) * quantized->depth);

    quantized->quantized_weights = malloc(sizeof(int8_t**) * quantized->depth);
    quantized->weight_scales = malloc(sizeof(float*) * quantized->depth);
    quantized->quantized_biases = malloc(sizeof(int8_t*) * quantized->depth);
    quantized->bias_scales = malloc(sizeof(float) * quantized->depth);

    // Load quantized weights and biases
    for (uint32_t layer = 1; layer < quantized->depth; layer++) {
        uint32_t curr_width = quantized->width[layer];
        uint32_t prev_width = quantized->width[layer - 1];

        quantized->quantized_weights[layer] = malloc(sizeof(int8_t*) * curr_width);
        quantized->weight_scales[layer] = malloc(sizeof(float) * curr_width);
        quantized->quantized_biases[layer] = malloc(sizeof(int8_t) * curr_width);

        for (uint32_t neuron = 0; neuron < curr_width; neuron++) {
            // Read weight scale
            if (fscanf(file, "%f\n", &quantized->weight_scales[layer][neuron]) != 1) {
                fprintf(stderr, "Error reading weight scale for layer %u neuron %u\n", layer, neuron);
                goto error;
            }

            // Read quantized weights
            quantized->quantized_weights[layer][neuron] = malloc(sizeof(int8_t) * prev_width);
            for (uint32_t w = 0; w < prev_width; w++) {
                int8_t weight;
                if (fscanf(file, "%hhd\n", &weight) != 1) {
                    fprintf(stderr, "Error reading weight for layer %u neuron %u weight %u\n", layer, neuron, w);
                    goto error;
                }
                quantized->quantized_weights[layer][neuron][w] = weight;
            }
        }

        // Read bias scale
        if (fscanf(file, "%f\n", &quantized->bias_scales[layer]) != 1) {
            fprintf(stderr, "Error reading bias scale for layer %u\n", layer);
            goto error;
        }

        // Read quantized biases
        for (uint32_t neuron = 0; neuron < curr_width; neuron++) {
            int8_t bias;
            if (fscanf(file, "%hhd\n", &bias) != 1) {
                fprintf(stderr, "Error reading bias for layer %u neuron %u\n", layer, neuron);
                goto error;
            }
            quantized->quantized_biases[layer][neuron] = bias;
        }
    }

    fclose(file);
    return network;

error:
    fclose(file);
    nn_free(network); // This will free quantized_network too
    return NULL;
}

float activate(float value, int activation_type) {
	// Bounds check to prevent invalid access
	if (activation_type < 0 || activation_type >= (int)(sizeof(activation_function)/sizeof(activation_function[0]))) {
		return activation_function[ACTIVATION_FUNCTION_TYPE_NONE](value, false);
	}
	return activation_function[activation_type](value, false);
}

uint32_t nn_version(void)
{
	return (NN_VERSION_MAJOR << 24) | (NN_VERSION_MINOR << 16) | (NN_VERSION_PATCH << 8) | NN_VERSION_BUILD;
}
