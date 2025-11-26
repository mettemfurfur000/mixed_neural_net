#ifndef DATASET_H
#define DATASET_H 1

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
dataset *load_csv_dataset(const char *filepath, int feature_size, int label_size, int skip_header);

// Dataset utilities
void normalize_dataset(dataset *ds, float min_val, float max_val);
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

#endif
