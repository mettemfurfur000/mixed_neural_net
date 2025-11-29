#ifndef DATASET_H
#define DATASET_H 1

#include <stdbool.h>
#include <stddef.h>

typedef struct
{
    float *features;  // Input features (flattened)
    float *label;     // Output label(s)
    int feature_size; // Number of features
    int label_size;   // Number of labels
} data_sample;

typedef struct
{
    data_sample *samples;
    int num_samples;
    int feature_size;
    int label_size;
} dataset;

// Dataset creation and destruction
dataset *create_dataset(int num_samples, int feature_size, int label_size);
void free_dataset(dataset *ds);

// CSV parsing
// label_position: 0 = labels at end (default), 1 = labels at beginning
dataset *load_csv_dataset(const char *filepath, int feature_size, int label_size, bool skip_header);
dataset *load_csv_dataset_with_label_pos(const char *filepath, int feature_size, int label_size, bool skip_header,
                                         int label_position);

// Dataset utilities
void normalize_dataset(dataset *ds, float min_val, float max_val);
void normalize_labels(dataset *ds, float min_val, float max_val);
void print_dataset_info(dataset *ds);
void print_sample(data_sample *sample);

// Train/test split
typedef struct
{
    dataset *train;
    dataset *test;
} train_test_split;

train_test_split split_dataset(dataset *ds, double train_ratio);
void free_train_test_split(train_test_split *split);

data_sample *dataset_random_sample(dataset *ds);

// Expand label size for a dataset (used for one-hot encoding)
// This reallocates all labels to the new size and initializes new elements to 0
int expand_dataset_labels(dataset *ds, int new_label_size);

// Conversion functions for multi-class classification
// Convert a single label (0-9) to one-hot encoding (10 outputs)
void label_to_one_hot(float label, float *output, int num_classes);

// Prediction result structure for multi-class classification
typedef struct
{
    int predicted_digit;  // The predicted digit (0-9)
    float confidence;     // Confidence percentage (0-100)
    float *class_outputs; // Raw output values for each class
} prediction_result;

// Convert network outputs to a prediction result with digit and confidence
prediction_result *network_outputs_to_prediction(float *outputs, int num_outputs);
void free_prediction_result(prediction_result *pred);

#endif
