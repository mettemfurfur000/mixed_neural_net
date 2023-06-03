#ifndef DATA_MANAGEMENT_H
#define DATA_MANAGEMENT_H 1
#include "neurostructs.h"

layer *alloc_layer()
{
    return (layer *)calloc(1, sizeof(layer));
}

#endif