#include "../include/funcs.h"
#include "../include/nn.h"

int main()
{
    // a test program for the neural network library

    neural_func_list funcs;
    init_neural_func_list(&funcs);

    // create layers
    layer *input_layer = create_layer(3, funcs.funcs[FUNC_LINE]);
    layer *hidden_layer = create_layer(5, funcs.funcs[FUNC_RELU]);
    layer *output_layer = create_layer(2, funcs.funcs[FUNC_LINE]);

    // create neural network
    layer layers_array[3] = {*input_layer, *hidden_layer, *output_layer};
    neural_net *net = create_neural_net(3, layers_array);

    // sample input and expected output
    float inputs[3] = {0.5f, -0.2f, 0.1f};
    float expected_outputs[2] = {0.7f, 0.3f};

    // train the network
    for (int epoch = 0; epoch < 150; epoch++)
    {
        train_neural_net(net, inputs, expected_outputs, 0.01f);
    }

    // find better activation functions
    find_better_function(net, &funcs, inputs, expected_outputs, 0.1f);

    print_neural_net(net);

    // cleanup

    free_neural_net(net);

    return 0;
}