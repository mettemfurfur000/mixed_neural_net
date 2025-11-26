#include "../include/io.h"
#include "../include/funcs.h"
#include "../include/general.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC_NUMBER 0x4E4E5700 // "NNW\0"
#define FILE_VERSION 1

int get_activation_id(float (*f)(float))
{
    if (f == line)
        return FUNC_LINE;
    else if (f == step)
        return FUNC_STEP;
    else if (f == relu)
        return FUNC_RELU;
    else if (f == sigmoid)
        return FUNC_SIGMOID;
    else if (f == softsign)
        return FUNC_SOFTSIGN;
    else
        return -1; // Unknown function
}

neural_func get_activation_from_id(int id, neural_func_list *funcs)
{
    if (!funcs || id < 0 || id >= funcs->size)
    {
        // Return a default function (line)
        neural_func default_func;
        default_func.f = line;
        default_func.f_d = line_d;
        return default_func;
    }

    return funcs->funcs[id];
}

int save_neural_net(const char *filepath, neural_net *net, neural_func_list *funcs)
{
    if (!filepath || !net)
    {
        fprintf(stderr, "Invalid filepath or network\n");
        return 0;
    }

    FILE *file = fopen(filepath, "wb");
    if (!file)
    {
        fprintf(stderr, "Failed to open file for writing: %s\n", filepath);
        return 0;
    }

    // Write header
    network_header header;
    header.magic = MAGIC_NUMBER;
    header.version = FILE_VERSION;
    header.num_layers = net->size;

    if (fwrite(&header.magic, sizeof(int), 1, file) != 1 || fwrite(&header.version, sizeof(int), 1, file) != 1 ||
        fwrite(&header.num_layers, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "Failed to write header\n");
        fclose(file);
        return 0;
    }

    // Write layer sizes
    for (u32 i = 0; i < net->size; i++)
    {
        int layer_size = net->layers[i].size;
        if (fwrite(&layer_size, sizeof(int), 1, file) != 1)
        {
            fprintf(stderr, "Failed to write layer size\n");
            fclose(file);
            return 0;
        }
    }

    // Write all neuron biases and activation function IDs
    for (u32 i = 0; i < net->size; i++)
    {
        layer *l = &net->layers[i];
        for (int j = 0; j < l->size; j++)
        {
            // Write bias
            if (fwrite(&l->neurons[j].bias, sizeof(float), 1, file) != 1)
            {
                fprintf(stderr, "Failed to write bias\n");
                fclose(file);
                return 0;
            }

            // Write activation function ID
            int act_id = get_activation_id(l->neurons[j].activation.f);
            if (fwrite(&act_id, sizeof(int), 1, file) != 1)
            {
                fprintf(stderr, "Failed to write activation ID\n");
                fclose(file);
                return 0;
            }
        }
    }

    // Write all weights
    for (u32 i = 0; i < net->size - 1; i++)
    {
        layer *current = &net->layers[i];
        layer *next = &net->layers[i + 1];

        for (int j = 0; j < current->size; j++)
        {
            if (fwrite(current->weights[j], sizeof(float), next->size, file) != (size_t)next->size)
            {
                fprintf(stderr, "Failed to write weights\n");
                fclose(file);
                return 0;
            }
        }
    }

    fclose(file);
    return 1;
}

neural_net *load_neural_net(const char *filepath, neural_func_list *funcs)
{
    if (!filepath)
    {
        fprintf(stderr, "Invalid filepath\n");
        return NULL;
    }

    FILE *file = fopen(filepath, "rb");
    if (!file)
    {
        fprintf(stderr, "Failed to open file for reading: %s\n", filepath);
        return NULL;
    }

    // Read header
    int magic, version, num_layers;
    if (fread(&magic, sizeof(int), 1, file) != 1 || fread(&version, sizeof(int), 1, file) != 1 ||
        fread(&num_layers, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "Failed to read header\n");
        fclose(file);
        return NULL;
    }

    // Verify magic number
    if (magic != MAGIC_NUMBER)
    {
        fprintf(stderr, "Invalid file format: wrong magic number\n");
        fclose(file);
        return NULL;
    }

    // Verify sizes
    if (num_layers > 1024)
    {
        fprintf(stderr, "Too many layers: %d\n", num_layers);
        fclose(file);
        return NULL;
    }

    // Read layer sizes
    int *layer_sizes = calloc(num_layers, sizeof(int));
    assert(layer_sizes);

    for (u32 i = 0; i < num_layers; i++)
    {
        if (fread(&layer_sizes[i], sizeof(int), 1, file) != 1)
        {
            fprintf(stderr, "Failed to read layer size\n");
            exit(-1);
        }
    }

    // Create layers
    layer *layers = calloc(num_layers, sizeof(layer));
    assert(layers);

    // Initialize each layer
    for (u32 i = 0; i < num_layers; i++)
    {
        layers[i].size = layer_sizes[i];
        layers[i].neurons = calloc(layer_sizes[i], sizeof(neuron));
        assert(layers[i].neurons);
        layers[i].next = NULL;
        layers[i].prev = NULL;
        layers[i].weights = NULL;

        // Initialize neuron fields
        for (int j = 0; j < layer_sizes[i]; j++)
        {
            layers[i].neurons[j].output = 0.0f;
            layers[i].neurons[j].delta = 0.0f;
        }
    }

    // Read neuron biases and activation IDs
    for (u32 i = 0; i < num_layers; i++)
    {
        for (int j = 0; j < layers[i].size; j++)
        {
            // Read bias
            if (fread(&layers[i].neurons[j].bias, sizeof(float), 1, file) != 1)
            {
                fprintf(stderr, "Failed to read bias\n");
                exit(-1);
            }

            // Read and set activation function
            int act_id;
            if (fread(&act_id, sizeof(int), 1, file) != 1)
            {
                fprintf(stderr, "Failed to read activation ID\n");
                exit(-1);
            }

            layers[i].neurons[j].activation = get_activation_from_id(act_id, funcs);
        }
    }

    // Create network and initialize weights
    neural_net *net = calloc(1, sizeof(neural_net));
    assert(net);

    net->size = num_layers;
    net->layers = layers;

    // Initialize weights matrices
    for (u32 i = 0; i < num_layers - 1; i++)
    {
        layer *current = &net->layers[i];
        layer *next = &net->layers[i + 1];

        current->weights = calloc(current->size, sizeof(float *));
        assert(current->weights);

        for (int j = 0; j < current->size; j++)
        {
            current->weights[j] = calloc(next->size, sizeof(float));
            assert(current->weights);

            // Read weights
            if (fread(current->weights[j], sizeof(float), next->size, file) != (size_t)next->size)
            {
                fprintf(stderr, "Failed to read weights\n");
                exit(-1);
            }
        }

        // Link layers
        current->next = next;
        next->prev = current;
    }

    SAFE_FREE(layer_sizes);
    fclose(file);

    return net;
}
