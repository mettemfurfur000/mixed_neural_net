#include "../include/funcs.h"
#include <assert.h>
#include <math.h>
//
float line(float x)
{
    return x;
}

float line_d(float x)
{
    return 1;
}

//
float step(float x)
{
    return x < 0 ? 1 : 0;
}

float step_d(float x)
{
    return x < 0 ? 1 : 0;
}

//
float relu(float x)
{
    return x > 0 ? x : 0;
}

float relu_d(float x)
{
    return x > 0 ? 1 : 0;
}

//
float sigmoid(float x)
{
    return 1 / (1 + pow(M_E, -x));
}

float sigmoid_d(float x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}
//
float softsign(float x)
{
    return x / (1 + fabs(x));
}

float softsign_d(float x)
{
    return 1 / ((1 + fabs(x)) * (1 + fabs(x)));
}

void init_neural_func_list(neural_func_list *list)
{
    if (!list)
        return;

    list->size = 5;
    list->funcs = calloc(list->size, sizeof(neural_func));
    assert(list->funcs);

    list->funcs[FUNC_LINE].f = line;
    list->funcs[FUNC_LINE].f_d = line_d;

    list->funcs[FUNC_STEP].f = step;
    list->funcs[FUNC_STEP].f_d = step_d;

    list->funcs[FUNC_RELU].f = relu;
    list->funcs[FUNC_RELU].f_d = relu_d;

    list->funcs[FUNC_SIGMOID].f = sigmoid;
    list->funcs[FUNC_SIGMOID].f_d = sigmoid_d;

    list->funcs[FUNC_SOFTSIGN].f = softsign;
    list->funcs[FUNC_SOFTSIGN].f_d = softsign_d;
}