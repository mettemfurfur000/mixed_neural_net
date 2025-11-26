# mixed_neural_net

A modular neural network library written in pure C with support for adaptive activation function mutation, dataset handling, and network serialization.

## Features

### Core Neural Network (`nn.h`, `nn.c`)
- **Layer creation and management**: Build networks with arbitrary layer sizes
- **Forward pass**: Propagate inputs through the network
- **Backward pass**: Compute gradients using backpropagation
- **Weight updates**: Gradient descent optimization
- **Activation functions**: Support for line, step, ReLU, sigmoid, and softsign
- **Adaptive mutation**: Dynamically change activation functions to overcome training plateaus
  - Targets low-contribution neurons
  - Stochastic mutations with configurable probability
  - Tests candidate functions before commitment

### Dataset Management (`dataset.h`, `dataset.c`)
- **Flexible dataset structures**: Support arbitrary feature and label sizes
- **CSV parsing**: Load data from CSV files with configurable headers
- **Data normalization**: Min-max scaling to arbitrary ranges
- **Train/test splitting**: Create validation sets with specified ratios
- **Memory efficient**: Handles large datasets with proper memory management

### Network I/O (`io.h`, `io.c`)
- **Binary serialization**: Save trained networks to `.nnw` files
- **Network loading**: Restore networks with all weights, biases, and activation functions
- **Format verification**: Magic number checking for file integrity
- **Version support**: Future-proof format for backward compatibility

## Building

```bash
make clean
make all
```

This creates an executable at `build/nn` that demonstrates all functionality.

## Project Structure

```
.
├── include/
│   ├── nn.h           # Core neural network API
│   ├── funcs.h        # Activation functions
│   ├── dataset.h      # Dataset structures and functions
│   └── io.h           # Serialization API
├── src/
│   ├── nn.c           # Neural network implementation
│   ├── funcs.c        # Activation function implementations
│   ├── dataset.c      # Dataset handling
│   └── io.c           # I/O implementation
├── mains/
│   └── base.c         # Example program with all features
├── obj/               # Compiled object files
├── build/             # Final executable
└── makefile           # Build configuration
```

## Usage Examples

### Creating and Training a Network

```c
#include "nn.h"
#include "funcs.h"

neural_func_list funcs;
init_neural_func_list(&funcs);

// Create layers
layer *input = create_layer(784, funcs.funcs[FUNC_LINE]);      // 28x28 MNIST images
layer *hidden = create_layer(128, funcs.funcs[FUNC_RELU]);
layer *output = create_layer(10, funcs.funcs[FUNC_SIGMOID]);

layer layers[3] = {*input, *hidden, *output};
neural_net *net = create_neural_net(3, layers);

// Train on a sample
float inputs[784] = {...};
float targets[10] = {...};

for (int epoch = 0; epoch < 1000; epoch++) {
    train_neural_net(net, inputs, targets, 0.01f);
    
    // Occasionally mutate activation functions
    if (epoch % 100 == 0) {
        find_better_function(net, &funcs, inputs, targets, 0.05f);
    }
}
```

### Loading Datasets

```c
#include "dataset.h"

// Load from CSV (features + label columns)
dataset *ds = load_csv_dataset("mnist_data.csv", 784, 10, 1);  // Skip header

if (ds) {
    printf("Loaded %d samples\n", ds->num_samples);
    
    // Normalize to [0, 1]
    normalize_dataset(ds, 0.0f, 1.0f);
    
    // Split into train/test
    train_test_split split = split_dataset(ds, 0.8f);
    
    // Use split.train and split.test...
    
    free_train_test_split(&split);
    free_dataset(ds);
}
```

### Saving and Loading Networks

```c
#include "io.h"

// Save trained network
save_neural_net("my_model.nnw", net, &funcs);

// Later, load it back
neural_net *loaded = load_neural_net("my_model.nnw", &funcs);

// Use loaded network...

free_neural_net(loaded);
```

## Activation Functions

- **Line**: Identity function (f(x) = x)
- **Step**: Binary activation (f(x) = x < 0 ? 1 : 0)
- **ReLU**: Rectified linear unit (f(x) = x > 0 ? x : 0)
- **Sigmoid**: Logistic function (f(x) = 1 / (1 + e^-x))
- **Softsign**: Smooth activation (f(x) = x / (1 + |x|))

## Adaptive Mutation System

The network supports intelligent mutation of activation functions during training:

1. **Contribution Analysis**: Evaluates each neuron's impact on network output
2. **Selective Mutation**: Targets neurons with low contribution or random selection
3. **Fitness Testing**: Tests candidate functions before applying changes
4. **Stochastic Element**: Configurable mutation probability prevents overfitting to local optima

This allows networks to escape training plateaus by exploring alternative activation functions.

## Memory Management

All functions follow consistent memory management patterns:
- `create_*` functions allocate memory
- `free_*` functions deallocate memory
- Always match `create_` with `free_` calls

## Building with MNIST Dataset

To train on MNIST:
1. Download the dataset in CSV format (784 pixel features + 10 class labels)
2. Use `load_csv_dataset("mnist.csv", 784, 10, 1)` to load
3. Split and normalize before training

## Notes

- Compiled with `-Os` optimization flag
- Static analysis enabled (`-fanalyzer`) for debugging
- Math library linked (`-lm`)
- Windows-compatible (tested on MSYS2)

## Compilation Notes

- **No external dependencies**: Pure C implementation
- **C99 standard**: Compatible with modern C compilers
- **Thread-safe**: Each network instance is independent
- **No C++**: Pure C code maintained throughout
