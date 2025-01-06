/*
 * Neural Network library
 * Copyright (c) 2019-2024 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdint.h>
#include "nn.h"

int main(void)
{
    nn_t *nn;

    nn = nn_load("model.txt");
    if (NULL == nn) {
        printf("Error: Missing or invalid model file.\n");
        return 1;
    }
    printf("Layer\tType\tWidth\tActivation\tBias\n");
    for (int i = 0; i < nn->depth; i++) {
        printf("%d\t%s\t%d\t%d\t%.5f\n", i, "dense", nn->width[i], nn->activation[i], nn->bias[i] / 128.0); // Convert fixed-point to float for display
    }
    return 0;
}