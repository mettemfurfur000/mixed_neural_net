#ifndef FUNCS_H
#define FUNCS_H

#include <math.h>
#include <stdlib.h>

typedef struct
{
    float (*f)(float);   // Pointer to the activation function
    float (*f_d)(float); // Pointer to the derivative of the activation function
} neural_func;

typedef struct
{
    int size;
    neural_func *funcs;
} neural_func_list;

#define RAND_FLOAT (rand() / (float)RAND_MAX)

#define FUNC_LINE 0
#define FUNC_STEP 1
#define FUNC_RELU 2
#define FUNC_SIGMOID 3
#define FUNC_SOFTSIGN 4

float line(float x);
float line_d(float x);

float step(float x);
float step_d(float x);

float relu(float x);
float relu_d(float x);

float sigmoid(float x);
float sigmoid_d(float x);

float softsign(float x);
float softsign_d(float x);

void init_neural_func_list(neural_func_list *list);

#endif