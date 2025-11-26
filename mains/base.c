#include "../include/dataset.h"
#include "../include/funcs.h"
#include "../include/general.h"
#include "../include/io.h"
#include "../include/nn.h"

#include <stdio.h>
#include <time.h>

int main()
{
    srand((unsigned int)time(NULL));

    neural_func_list funcs;
    init_neural_func_list(&funcs);

    // ============ DEMO 1: Basic training with simple data ============
    printf("=== DEMO 1: Basic Network Training ===\n");

    // create layers
    layer *input_layer = create_layer(3, funcs.funcs[FUNC_LINE]);
    layer *hidden_layer = create_layer(5, funcs.funcs[FUNC_RELU]);
    layer *output_layer = create_layer(2, funcs.funcs[FUNC_SIGMOID]);

    // create neural network
    layer layers_array[3] = {*input_layer, *hidden_layer, *output_layer};
    neural_net *net = create_neural_net(3, layers_array);

    // sample input and expected output
    float inputs[3] = {0.5f, -0.2f, 0.1f};
    float expected_outputs[2] = {0.7f, 0.3f};

    // train the network
    printf("Training for 500 epochs...\n");
    for (int epoch = 0; epoch < 500; epoch++)
    {
        train_neural_net(net, inputs, expected_outputs, 0.01f);

        // Apply mutations occasionally
        if (epoch % 50 == 0)
        {
            find_better_function(net, &funcs, inputs, expected_outputs, 0.1f);
        }
    }

    printf("Training complete!\n");
    print_neural_net(net);

    printf("Fitness: %f\n", neural_net_fitness(net, inputs, expected_outputs));

    // ============ DEMO 2: Save and Load Network ============
    printf("\n=== DEMO 2: Save and Load Network ===\n");

    const char *model_path = "trained_model.nnw";
    printf("Saving network to: %s\n", model_path);
    if (save_neural_net(model_path, net, &funcs))
    {
        printf("Network saved successfully!\n");

        // Load the network back
        printf("Loading network from: %s\n", model_path);
        neural_net *loaded_net = load_neural_net(model_path, &funcs);
        if (loaded_net)
        {
            printf("Network loaded successfully!\n");

            // Test the loaded network
            forward_pass(loaded_net, inputs);
            layer *output = &loaded_net->layers[loaded_net->size - 1];
            printf("Loaded network output: [%.4f, %.4f]\n", output->neurons[0].output, output->neurons[1].output);

            free_neural_net(loaded_net);
        }
        else
        {
            printf("Failed to load network\n");
        }
    }
    else
    {
        printf("Failed to save network\n");
    }

    // ============ DEMO 3: Dataset Creation and Normalization ============
    printf("\n=== DEMO 3: Dataset Creation ===\n");

    int num_samples = 100;
    int feature_size = 4;
    int label_size = 1;

    dataset *ds = create_dataset(num_samples, feature_size, label_size);
    if (ds)
    {
        printf("Created dataset with %d samples\n", ds->num_samples);
        printf("Features per sample: %d, Labels per sample: %d\n", ds->feature_size, ds->label_size);

        // Fill with dummy data
        for (u32 i = 0; i < num_samples; i++)
        {
            for (int j = 0; j < feature_size; j++)
                ds->samples[i].features[j] = (rand() / (float)RAND_MAX) * 255.0f;
            ds->samples[i].label[0] = (rand() / (float)RAND_MAX);
        }

        printf("Before normalization: first sample features [%.2f, %.2f, %.2f, %.2f]\n", ds->samples[0].features[0],
               ds->samples[0].features[1], ds->samples[0].features[2], ds->samples[0].features[3]);

        // Normalize to [0, 1]
        normalize_dataset(ds, 0.0f, 1.0f);

        printf("After normalization: first sample features [%.4f, %.4f, %.4f, %.4f]\n", ds->samples[0].features[0],
               ds->samples[0].features[1], ds->samples[0].features[2], ds->samples[0].features[3]);

        print_dataset_info(ds);
        printf("Sample 0:\n");
        print_sample(&ds->samples[0]);

        free_dataset(ds);
    }

    // ============ DEMO 4: Train/Test Split ============
    printf("\n=== DEMO 4: Train/Test Split ===\n");

    dataset *full_ds = create_dataset(100, 4, 1);
    if (full_ds)
    {
        // Fill with data
        for (u32 i = 0; i < 100; i++)
        {
            for (int j = 0; j < 4; j++)
                full_ds->samples[i].features[j] = (rand() / (float)RAND_MAX);
            full_ds->samples[i].label[0] = (rand() / (float)RAND_MAX);
        }

        train_test_split split = split_dataset(full_ds, 0.8f);
        if (split.train && split.test)
        {
            printf("Full dataset: %d samples\n", full_ds->num_samples);
            printf("Training set: %d samples (80%%)\n", split.train->num_samples);
            printf("Test set: %d samples (20%%)\n", split.test->num_samples);

            free_train_test_split(&split);
        }

        free_dataset(full_ds);
    }

    // cleanup
    free_neural_net(net);

    // Free the original layers created before the network
    free_layer(input_layer);
    free_layer(hidden_layer);
    free_layer(output_layer);

    printf("\n=== All demos completed! ===\n");

    return 0;
}