#include <math.h>
#include <stdlib.h>

#define mth_E 2.7182818284590452354

struct activation_function
{
    float (*f)(float);
    float (*f_d)(float);
};

struct neural_func_list
{
    int size;
    struct activation_function *act;
};

float init_weight() { return (rand()/(float)RAND_MAX); }

float line(float x);
float line_d(float x);

float step(float x);
float step_d(float x);

float relu(float x);
float relu_d(float x);

float sigmoid(float x);
float sigmoid_d(float x);

float Q_rsqrt(float number);
float s_rsqrt_d(float x);

float softsign(float x);
float softsign_d(float x);