#ifndef NN_H
#define NN_H 1

#include "funcs.h"

typedef struct neuron
{
    float bias;             // Bias for the neuron
    float output;           // Output of the neuron
    neural_func activation; // Activation function for the neuron
    float delta;            // Delta value for backpropagation
} neuron;

typedef struct layer layer;

typedef struct layer
{
    int size;
    neuron *neurons; // Array of neurons in the layer

    // might point to inputs / outputs
    layer *next;
    layer *prev;

    float **weights; // Weights connecting to the next layer
} layer;

typedef struct
{
    int size;
    layer *layers;
} neural_net;

// Function prototypes
layer *create_layer(int size, neural_func default_func);
void free_layer(layer *l);

neural_net *create_neural_net(int size, layer *layers);
void free_neural_net(neural_net *net);

void forward_pass(neural_net *net, float *inputs);
void backward_pass(neural_net *net, float *expected_outputs);
void update_weights(neural_net *net, float learning_rate);
void train_neural_net(neural_net *net, float *inputs, float *expected_outputs, float learning_rate);

void print_neural_net(neural_net *net);
float neural_net_fitness(neural_net *net, float *inputs, float *expected_outputs);

// Function to find a better activation function - will change the function to a one that fits better right now with a
// certain chance
void find_better_function(neural_net *net, neural_func_list *funcs, float *inputs, float *expected_outputs,
                          float mutation_rate);

#endif