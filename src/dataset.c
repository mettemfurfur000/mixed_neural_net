#include "../include/dataset.h"
#include "../include/general.h"

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

dataset *create_dataset(int num_samples, int feature_size, int label_size)
{
    dataset *ds = calloc(1, sizeof(dataset));
    assert(ds);
    assert(num_samples != 0);
    ds->num_samples = num_samples;
    ds->feature_size = feature_size;
    ds->label_size = label_size;

    ds->samples = calloc(num_samples, sizeof(data_sample));
    assert(ds->samples);

    for (u32 i = 0; i < num_samples; i++)
    {
        ds->samples[i].features = calloc(feature_size, sizeof(float));
        ds->samples[i].label = calloc(label_size, sizeof(float));

        assert(ds->samples[i].features && ds->samples[i].label);

        ds->samples[i].feature_size = feature_size;
        ds->samples[i].label_size = label_size;
    }

    return ds;
}

void free_dataset(dataset *ds)
{
    if (ds)
    {
        for (u32 i = 0; i < ds->num_samples; i++)
        {
            SAFE_FREE(ds->samples[i].features)
            SAFE_FREE(ds->samples[i].label)
        }
        SAFE_FREE(ds->samples);
        SAFE_FREE(ds);
    }
}

// Parse a single line of CSV - handles quoted fields and commas
static int parse_csv_line(char *line, float *values, int max_values)
{
    int count = 0;
    char *ptr = line;
    char buffer[256];
    int buf_idx = 0;
    int in_quotes = 0;

    while (*ptr && count < max_values)
    {
        if (*ptr == '"')
        {
            in_quotes = !in_quotes;
            ptr++;
        }
        else if (*ptr == ',' && !in_quotes)
        {
            buffer[buf_idx] = '\0';
            values[count++] = atof(buffer);
            buf_idx = 0;
            ptr++;
        }
        else if (*ptr == '\n' || *ptr == '\r')
        {
            break;
        }
        else
        {
            buffer[buf_idx++] = *ptr;
            ptr++;
        }

        if (buf_idx >= 255)
        {
            fprintf(stderr, "CSV field too long\n");
            return -1;
        }
    }

    // Handle last field
    if (buf_idx > 0 || (*ptr == ',' && count < max_values))
    {
        buffer[buf_idx] = '\0';
        values[count++] = atof(buffer);
    }

    return count;
}

dataset *load_csv_dataset(const char *filepath, int feature_size, int label_size, int skip_header)
{
    FILE *file = fopen(filepath, "r");
    if (!file)
    {
        fprintf(stderr, "Failed to open file: %s\n", filepath);
        return NULL;
    }

    // Count lines (excluding header)
    int line_count = 0;
    char buffer[8192];
    if (skip_header && fgets(buffer, sizeof(buffer), file) == NULL)
    {
        fprintf(stderr, "Failed to skip header\n");
        fclose(file);
        return NULL;
    }

    while (fgets(buffer, sizeof(buffer), file))
    {
        // Skip empty lines
        int is_empty = 1;
        for (u32 i = 0; buffer[i]; i++)
        {
            if (!isspace(buffer[i]))
            {
                is_empty = 0;
                break;
            }
        }
        if (!is_empty)
            line_count++;
    }

    // Create dataset
    dataset *ds = create_dataset(line_count, feature_size, label_size);
    if (!ds)
    {
        fclose(file);
        return NULL;
    }

    // Reset file pointer
    rewind(file);

    // Skip header again
    if (skip_header)
        fgets(buffer, sizeof(buffer), file);

    // Read data
    float *values = calloc(feature_size + label_size, sizeof(float));
    assert(values);

    int sample_idx = 0;
    while (fgets(buffer, sizeof(buffer), file) && sample_idx < line_count)
    {
        // Skip empty lines
        int is_empty = 1;
        for (u32 i = 0; buffer[i]; i++)
        {
            if (!isspace(buffer[i]))
            {
                is_empty = 0;
                break;
            }
        }
        if (is_empty)
            continue;

        // Parse line
        int parsed = parse_csv_line(buffer, values, feature_size + label_size);
        if (parsed != feature_size + label_size)
        {
            fprintf(stderr, "Warning: Line %d has %d values, expected %d\n", sample_idx + (skip_header ? 2 : 1), parsed,
                    feature_size + label_size);
            continue;
        }

        // Copy features and labels
        memcpy(ds->samples[sample_idx].features, values, feature_size * sizeof(float));
        memcpy(ds->samples[sample_idx].label, values + feature_size, label_size * sizeof(float));

        sample_idx++;
    }

    // Trim dataset if fewer samples were read
    if (sample_idx < line_count)
    {
        ds->num_samples = sample_idx;
    }

    SAFE_FREE(values);
    fclose(file);

    return ds;
}

void normalize_dataset(dataset *ds, float min_val, float max_val)
{
    if (!ds || ds->num_samples == 0)
        return;

    // Find min and max for each feature
    float *feature_min = calloc(ds->feature_size, sizeof(float));
    float *feature_max = calloc(ds->feature_size, sizeof(float));

    assert(feature_min && feature_max);

    // Initialize with first sample
    for (u32 i = 0; i < ds->feature_size; i++)
    {
        feature_min[i] = ds->samples[0].features[i];
        feature_max[i] = ds->samples[0].features[i];
    }

    // Find min/max
    for (u32 i = 0; i < ds->num_samples; i++)
    {
        for (int j = 0; j < ds->feature_size; j++)
        {
            if (ds->samples[i].features[j] < feature_min[j])
                feature_min[j] = ds->samples[i].features[j];
            if (ds->samples[i].features[j] > feature_max[j])
                feature_max[j] = ds->samples[i].features[j];
        }
    }

    // Normalize
    for (u32 i = 0; i < ds->num_samples; i++)
    {
        for (int j = 0; j < ds->feature_size; j++)
        {
            float range = feature_max[j] - feature_min[j];
            if (range > 0.0001f)
            {
                float normalized = (ds->samples[i].features[j] - feature_min[j]) / range;
                ds->samples[i].features[j] = normalized * (max_val - min_val) + min_val;
            }
        }
    }

    SAFE_FREE(feature_min);
    SAFE_FREE(feature_max);
}

void print_dataset_info(dataset *ds)
{
    if (!ds)
        return;

    printf("Dataset Info:\n");
    printf("  Samples: %d\n", ds->num_samples);
    printf("  Features per sample: %d\n", ds->feature_size);
    printf("  Labels per sample: %d\n", ds->label_size);
}

void print_sample(data_sample *sample)
{
    if (!sample)
        return;

    printf("Sample - Features: [");
    for (u32 i = 0; i < sample->feature_size; i++)
    {
        printf("%.4f", sample->features[i]);
        if (i < sample->feature_size - 1)
            printf(", ");
    }
    printf("], Labels: [");
    for (u32 i = 0; i < sample->label_size; i++)
    {
        printf("%.4f", sample->label[i]);
        if (i < sample->label_size - 1)
            printf(", ");
    }
    printf("]\n");
}

train_test_split split_dataset(dataset *ds, double train_ratio)
{
    assert(ds != NULL);
    train_test_split split = {NULL, NULL};

    if (!ds || train_ratio <= 0.0f || train_ratio >= 1.0f)
    {
        fprintf(stderr, "Invalid dataset or train ratio\n");
        return split;
    }

    u32 train_size = CLAMP((ds->num_samples * train_ratio), 0, ds->num_samples);
    u32 test_size = ds->num_samples - train_size;

    assert(train_size != 0 && test_size != 0);

    split.train = create_dataset(train_size, ds->feature_size, ds->label_size);
    split.test = create_dataset(test_size, ds->feature_size, ds->label_size);

    if (!split.train || !split.test)
    {
        fprintf(stderr, "Failed to create split datasets\n");
        free_dataset(split.train);
        free_dataset(split.test);
        split.train = NULL;
        split.test = NULL;
        return split;
    }

    // Copy samples
    for (u32 i = 0; i < train_size; i++)
    {
        memcpy(split.train->samples[i].features, ds->samples[i].features, ds->feature_size * sizeof(float));
        memcpy(split.train->samples[i].label, ds->samples[i].label, ds->label_size * sizeof(float));
    }

    for (u32 i = 0; i < test_size; i++)
    {
        memcpy(split.test->samples[i].features, ds->samples[train_size + i].features, ds->feature_size * sizeof(float));
        memcpy(split.test->samples[i].label, ds->samples[train_size + i].label, ds->label_size * sizeof(float));
    }

    return split;
}

void free_train_test_split(train_test_split *split)
{
    if (split)
    {
        free_dataset(split->train);
        free_dataset(split->test);
    }
}
