#include <math.h>
#include <stdlib.h>

#include "functions.h"
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
    return 1 / (1 + pow(mth_E, -x));
}

float sigmoid_d(float x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

// legend here
float Q_rsqrt(float number)
{
    long i;
    float x2, y;
    const float threehalfs = 1.5F;

    x2 = number * 0.5F;
    y = number;
    i = *(long *)&y;           // evil floating point bit level hacking
    i = 0x5f3759df - (i >> 1); // what the fuck?
    y = *(float *)&i;
    y = y * (threehalfs - (x2 * y * y)); // 1st iteration
//	y = y * (threehalfs - (x2 * y * y));   // 2nd iteration, this can be removed

    return y;
}

float s_rsqrt_d(float x)
{
    return pow((1 / sqrt(1 + pow(x, 2))), 3);
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