#include "../include/nn.h"

#include "../include/general.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define RAND_FLOAT (rand() / (float)RAND_MAX)

layer *create_layer(int size, neural_func default_func)
{
    layer *l = calloc(1, sizeof(layer));
    assert(l);

    l->size = size;
    l->neurons = calloc(size, sizeof(neuron));
    assert(l->neurons);

    for (u32 i = 0; i < size; i++)
    {
        l->neurons[i].bias = (RAND_FLOAT - 0.5f) * 2.0f; // Random bias between -1 and 1
        l->neurons[i].output = 0.0f;
        l->neurons[i].activation = default_func;
        l->neurons[i].delta = 0.0f;
    }

    l->next = NULL;
    l->prev = NULL;

    // Weights will be initialized after linking layers
    l->weights = NULL;

    return l;
}

void free_layer(layer *l)
{
    if (l)
    {
        if (l->weights)
        {
            for (u32 i = 0; i < l->size; i++)
            {
                SAFE_FREE(l->weights[i]);
            }
            SAFE_FREE(l->weights);
        }
        SAFE_FREE(l->neurons);
        SAFE_FREE(l);
    }
}

neural_net *create_neural_net(int size, layer *layers)
{
    assert(size > 0);
    assert(layers != NULL);

    neural_net *net = calloc(1, sizeof(neural_net));
    assert(net);

    // Don't copy layers - just use the provided array
    // The caller owns the layer array lifetime
    net->layers = layers;
    net->size = size;

    // Initialize weights between layers
    for (u32 i = 0; i < size - 1; i++)
    {
        layer *current = &net->layers[i];
        layer *next = &net->layers[i + 1];

        // Allocate weight matrix (current layer neurons x next layer neurons)
        current->weights = calloc(current->size, sizeof(float *));
        assert(current->weights);

        for (int j = 0; j < current->size; j++)
        {
            current->weights[j] = calloc(next->size, sizeof(float));
            assert(current->weights[j]);

            // Xavier initialization
            float limit = sqrtf(6.0f / (current->size + next->size));
            for (int k = 0; k < next->size; k++)
            {
                current->weights[j][k] = (RAND_FLOAT - 0.5f) * 2.0f * limit;
            }
        }

        // Link layers
        current->next = next;
        next->prev = current;
    }

    return net;
}

void free_neural_net(neural_net *net)
{
    if (!net)
        return;

    assert(net->layers != NULL);
    assert(net->size > 0);

    // Free weight matrices in each layer
    for (u32 i = 0; i < net->size - 1; i++)
    {
        layer *l = &net->layers[i];
        if (l->weights)
        {
            for (int j = 0; j < l->size; j++)
            {
                if (l->weights[j])
                {
                    SAFE_FREE(l->weights[j]);
                    l->weights[j] = NULL;
                }
            }
            SAFE_FREE(l->weights);
            l->weights = NULL;
        }
    }

    // NOTE: We don't free neurons or the layers array because they're managed
    // by the caller (usually stack-allocated). Only free the weights which we allocated.

    // Free network structure
    SAFE_FREE(net);
}

void forward_pass(neural_net *net, float *inputs)
{
    assert(net != NULL);
    assert(inputs != NULL);
    assert(net->layers != NULL);
    assert(net->size > 0);

    layer *input_layer = &net->layers[0];

    // Set inputs as outputs of first layer
    for (u32 i = 0; i < input_layer->size; i++)
    {
        input_layer->neurons[i].output = inputs[i];
    }

    // Forward through hidden and output layers
    for (int layer_idx = 1; layer_idx < net->size; layer_idx++)
    {
        layer *prev_layer = &net->layers[layer_idx - 1];
        layer *current_layer = &net->layers[layer_idx];

        for (int j = 0; j < current_layer->size; j++)
        {
            float sum = current_layer->neurons[j].bias;

            // Sum weighted inputs from previous layer
            for (u32 i = 0; i < prev_layer->size; i++)
            {
                sum += prev_layer->neurons[i].output * prev_layer->weights[i][j];
            }

            // Apply activation function
            current_layer->neurons[j].output = current_layer->neurons[j].activation.f(sum);
        }
    }
}

void backward_pass(neural_net *net, float *expected_outputs)
{
    assert(net != NULL);
    assert(expected_outputs != NULL);
    assert(net->layers != NULL);
    assert(net->size > 0);

    layer *output_layer = &net->layers[net->size - 1];

    // Calculate output layer deltas
    for (u32 i = 0; i < output_layer->size; i++)
    {
        float error = expected_outputs[i] - output_layer->neurons[i].output;
        float derivative = output_layer->neurons[i].activation.f_d(output_layer->neurons[i].output);
        output_layer->neurons[i].delta = error * derivative;
    }

    // Backpropagate through hidden layers
    for (int layer_idx = net->size - 2; layer_idx >= 0; layer_idx--)
    {
        layer *current_layer = &net->layers[layer_idx];
        layer *next_layer = &net->layers[layer_idx + 1];

        for (u32 i = 0; i < current_layer->size; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < next_layer->size; j++)
            {
                sum += next_layer->neurons[j].delta * current_layer->weights[i][j];
            }

            float derivative = current_layer->neurons[i].activation.f_d(current_layer->neurons[i].output);
            current_layer->neurons[i].delta = sum * derivative;
        }
    }
}

void update_weights(neural_net *net, float learning_rate)
{
    assert(net != NULL);
    assert(net->layers != NULL);
    assert(learning_rate > 0.0f);

    // Update weights and biases for all layers
    for (int layer_idx = 0; layer_idx < net->size - 1; layer_idx++)
    {
        layer *current_layer = &net->layers[layer_idx];
        layer *next_layer = &net->layers[layer_idx + 1];

        for (u32 i = 0; i < current_layer->size; i++)
        {
            for (int j = 0; j < next_layer->size; j++)
            {
                // Update weight: w = w + learning_rate * input * delta
                current_layer->weights[i][j] +=
                    learning_rate * current_layer->neurons[i].output * next_layer->neurons[j].delta;
            }
        }

        // Update biases
        for (int j = 0; j < next_layer->size; j++)
        {
            next_layer->neurons[j].bias += learning_rate * next_layer->neurons[j].delta;
        }
    }
}

void train_neural_net(neural_net *net, float *inputs, float *expected_outputs, float learning_rate)
{
    assert(net != NULL);
    assert(inputs != NULL);
    assert(expected_outputs != NULL);
    assert(learning_rate > 0.0f);

    forward_pass(net, inputs);
    backward_pass(net, expected_outputs);
    update_weights(net, learning_rate);
}

char *func_to_str(neural_func func)
{
    if (func.f == line)
        return "LINE";
    else if (func.f == step)
        return "STEP";
    else if (func.f == relu)
        return "RELU";
    else if (func.f == sigmoid)
        return "SIGMOID";
    else if (func.f == softsign)
        return "SOFTSIGN";
    else
        return "UNKNOWN";
}

void print_rgb_color(float r, float g, float b)
{
    printf("\x1b[38;2;%d;%d;%dm",         //
           CLAMP((int)(r * 255), 0, 255), //
           CLAMP((int)(g * 255), 0, 255), //
           CLAMP((int)(b * 255), 0, 255)  //
    );
}

#define AC_RESET "\x1b[0m"

void print_matrix_weights(float **weights, int rows, int cols)
{
    // print a grid of squares with colors based on weight values
    if (!weights)
        return;

    for (u32 i = 0; i < rows; i++)
    {
        printf("\t");
        for (int j = 0; j < cols; j++)
        {
            float weight = weights[i][j];
            float normalized = (weight + 1.0f) / 2.0f; // normalize to 0-1
            // map to color (red to green)
            float r = 1.0f - normalized;
            float g = normalized;
            print_rgb_color(r, g, 0);
            printf("[]");
        }
        printf("\n");
    }
    printf(AC_RESET);
}

void print_neural_net(neural_net *net)
{
    if (!net)
        return;

    printf("Neural Network with %d layers:\n", net->size);

    for (int layer_idx = 0; layer_idx < net->size; layer_idx++)
    {
        layer *l = &net->layers[layer_idx];
        printf("  Layer %d: %d neurons\n", layer_idx, l->size);

        for (u32 i = 0; i < l->size && i < 3; i++) // Print first 3 neurons
        {
            printf("    Neuron %d: bias=%.4f, output=%.4f, delta=%.4f, function=%s\n", i, l->neurons[i].bias,
                   l->neurons[i].output, l->neurons[i].delta, func_to_str(l->neurons[i].activation));
        }

        if (layer_idx < net->size - 1 && l->weights)
        {
            printf("    Weights to next layer: %d x %d matrix\n", l->size, net->layers[layer_idx + 1].size);
        }

        print_matrix_weights(l->weights, l->size, layer_idx < net->size - 1 ? net->layers[layer_idx + 1].size : 0);
    }
}

// Calculate neuron contribution to output - sum of absolute weight * previous layer output changes
static float calculate_neuron_contribution(layer *current_layer, layer *next_layer, int neuron_idx)
{
    if (!current_layer || !next_layer || !current_layer->weights)
        return 0.0f;

    float contribution = 0.0f;
    for (int j = 0; j < next_layer->size; j++)
    {
        contribution += fabsf(current_layer->weights[neuron_idx][j]) * fabsf(next_layer->neurons[j].delta);
    }
    return contribution;
}

// Find best activation function for a neuron based on a test pass
static neural_func find_best_activation(neural_net *net, neural_func_list *funcs, float *inputs,
                                        float *expected_outputs, int layer_idx, int neuron_idx,
                                        neural_func original_func)
{
    if (!net || !funcs || funcs->size == 0)
        return original_func;

    layer *target_layer = &net->layers[layer_idx];
    neuron original_neuron = target_layer->neurons[neuron_idx];

    float best_error = FLT_MAX;
    neural_func best_func = original_func;

    // Try each activation function
    for (int f_idx = 0; f_idx < funcs->size; f_idx++)
    {
        // Set the test activation function
        target_layer->neurons[neuron_idx].activation = funcs->funcs[f_idx];

        // Forward pass with test function
        forward_pass(net, inputs);
        backward_pass(net, expected_outputs);

        // Calculate error
        layer *output_layer = &net->layers[net->size - 1];
        float error = 0.0f;
        for (u32 i = 0; i < output_layer->size; i++)
        {
            float diff = expected_outputs[i] - output_layer->neurons[i].output;
            error += diff * diff;
        }

        if (error < best_error)
        {
            best_error = error;
            best_func = funcs->funcs[f_idx];
        }
    }

    // Restore original neuron state
    target_layer->neurons[neuron_idx] = original_neuron;

    return best_func;
}

void find_better_function(neural_net *net, neural_func_list *funcs, float *inputs, float *expected_outputs,
                          float mutation_rate)
{
    if (!net || !funcs || !mutation_rate || mutation_rate <= 0.0f)
        return;

    // Only consider hidden layers (skip input and output)
    for (int layer_idx = 1; layer_idx < net->size - 1; layer_idx++)
    {
        layer *current_layer = &net->layers[layer_idx];
        layer *next_layer = &net->layers[layer_idx + 1];

        for (int neuron_idx = 0; neuron_idx < current_layer->size; neuron_idx++)
        {
            // Two strategies for mutation:
            // 1. Mutate neurons with low contribution
            // 2. Random mutation with mutation_rate probability

            float contribution = calculate_neuron_contribution(current_layer, next_layer, neuron_idx);

            // Strategy: Mutate if random chance hits OR if contribution is low
            float random_chance = RAND_FLOAT;
            float contribution_threshold = 0.01f; // Neurons with very low contribution

            if (random_chance < mutation_rate || contribution < contribution_threshold)
            {
                neural_func new_func = find_best_activation(net, funcs, inputs, expected_outputs, layer_idx, neuron_idx,
                                                            current_layer->neurons[neuron_idx].activation);

                current_layer->neurons[neuron_idx].activation = new_func;
            }
        }
    }
}

float neural_net_fitness(neural_net *net, float *inputs, float *expected_outputs)
{
    if (!net || !inputs || !expected_outputs)
        return FLT_MAX;

    forward_pass(net, inputs);

    layer *output_layer = &net->layers[net->size - 1];
    float error = 0.0f;
    for (u32 i = 0; i < output_layer->size; i++)
    {
        float diff = expected_outputs[i] - output_layer->neurons[i].output;
        error += diff * diff;
    }

    return error;
}