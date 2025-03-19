#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "nn.h"
#include "data_prep.h"

// Unified evaluation function template
float evaluate_model(nn_predict_fn predict_fn, void *model, 
                    data_t *data, int output_size) {
    int num_samples = 0, correct = 0;
    for (int i = 0; i < data->num_rows; i++) {
        num_samples++;
        float *prediction = predict_fn(model, data->input[i]);
        int true_positive = 0, false_positive = 0;
        
        for (int j = 0; j < output_size; j++) {
            if (data->target[i][j] >= 0.5) {
                if (prediction[j] >= 0.5) true_positive++;
            } else {
                if (prediction[j] >= 0.5) false_positive++;
            }
        }
        
        if (true_positive == 1 && false_positive == 0) correct++;
    }
    return (correct * 100.0) / num_samples;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <model_file>\n", argv[0]);
        return 1;
    }

    // Check quantization flag
    FILE *model_file = fopen(argv[1], "r");
    if (!model_file) {
        fprintf(stderr, "Error opening model file: %s\n", argv[1]);
        return 1;
    }
    
    int is_quantized;
    if (fscanf(model_file, "%d", &is_quantized) != 1) {
        fclose(model_file);
        fprintf(stderr, "Invalid model file format\n");
        return 1;
    }
    fclose(model_file);

    // Load appropriate model type
    void *model = NULL;
    int input_size, output_size;
    nn_predict_fn predict_fn = NULL;

    if (is_quantized) {
        nn_quantized_t *qmodel = nn_load_quantized(argv[1]);
        if (!qmodel) {
            fprintf(stderr, "Failed to load quantized model\n");
            return 1;
        }
        model = qmodel;
        predict_fn = (nn_predict_fn)nn_predict_quantized;
        input_size = qmodel->original_network->width[0];
        output_size = qmodel->original_network->width[qmodel->original_network->depth - 1];
    } else {
        nn_t *fmodel = nn_load(argv[1]);
        if (!fmodel) {
            fprintf(stderr, "Failed to load float model\n");
            return 1;
        }
        model = fmodel;
        predict_fn = nn_predict;
        input_size = fmodel->width[0];
        output_size = fmodel->width[fmodel->depth - 1];
    }

    // Load and evaluate datasets
    data_t *data;
    const char *datasets[] = {"train.csv", "test.csv"};
    
    for (int d = 0; d < 2; d++) {
        data = data_load(datasets[d], input_size, output_size);
        if (!data) {
            fprintf(stderr, "Failed to load %s\n", datasets[d]);
            continue;
        }
        
        float accuracy = evaluate_model(predict_fn, model, data, output_size);
        printf("%s: %.2f%%\n", datasets[d], accuracy);
        data_free(data);
    }

    // Cleanup
    if (is_quantized) {
        nn_free_quantized((nn_quantized_t *)model);
    } else {
        nn_free((nn_t *)model);
    }

    return 0;
}