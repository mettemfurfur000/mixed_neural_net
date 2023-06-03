#ifndef NEUROSTRICTS_H
#define NEUROSTRICTS_H 1
#include "functions.h"

typedef struct layer
{
    int size;
    double* values;
    neural_func_list* f_list;
} layer;

typedef struct weights
{
    layer* in;
    layer* out;

    double** weights;
} weights;

typedef struct neural_net
{
    layer* layers;
    weights* w;
} neural_net;

#endif