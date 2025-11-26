#ifndef IO_H
#define IO_H 1

#include "nn.h"

typedef struct
{
    int magic;           // Magic number to verify file format (0x4E4E5700 = "NNW\0")
    int version;         // File version for future compatibility
    int num_layers;      // Number of layers
    int *layer_sizes;    // Size of each layer
    int *activation_ids; // Activation function ID for each neuron (flat array)
} network_header;

// Save and load neural networks
int save_neural_net(const char *filepath, neural_net *net, neural_func_list *funcs);
neural_net *load_neural_net(const char *filepath, neural_func_list *funcs);

// Helper to get activation function ID from function pointer
int get_activation_id(float (*f)(float));

// Helper to get activation function from ID
neural_func get_activation_from_id(int id, neural_func_list *funcs);

#endif
