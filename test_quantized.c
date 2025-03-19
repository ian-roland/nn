/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

 #include <stdio.h>
 #include <stdint.h>
 #include <stdlib.h>
 #include "nn.h"
 #include "data_prep.h"
 
 int main(void) {
	 nn_t *network = NULL;
	 data_t *data;
	 float *prediction;
	 int num_samples, correct, true_positive, false_positive;
 
	 // Load quantized model (now returns nn_t*)
	 network = nn_load_quantized("model_quantized.txt");
	 if (!network || !network->quantized_network) {
		 fprintf(stderr, "Error: Failed to load quantized model\n");
		 return 1;
	 }
 
	 // Get sizes from the original network
	 int input_size = network->width[0];
	 int output_size = network->width[network->depth - 1];
 
	 // Load training data
	 data = data_load("train.csv", input_size, output_size);
	 if (!data) {
		 fprintf(stderr, "Error: Failed to load training data\n");
		 nn_free(network);
		 return 1;
	 }
 
	 // Test training data
	 num_samples = 0;
	 correct = 0;
	 for (int i = 0; i < data->num_rows; i++) {
		 num_samples++;
		 prediction = nn_predict_quantized(network, data->input[i]);
		 true_positive = 0;
		 false_positive = 0;
 
		 for (int j = 0; j < output_size; j++) {
			 if (data->target[i][j] >= 0.5) {
				 if (prediction[j] >= 0.5)
					 true_positive++;
			 } else {
				 if (prediction[j] >= 0.5)
					 false_positive++;
			 }
		 }
 
		 if ((true_positive == 1) && (false_positive == 0))
			 correct++;
	 }
	 printf("Train: %d/%d = %2.2f%%\n", correct, num_samples, (correct * 100.0) / num_samples);
	 //data_free(data);
 
	 // Load test data
	 data = data_load("test.csv", input_size, output_size);
	 if (!data) {
		 fprintf(stderr, "Error: Failed to load test data\n");
		//nn_free(network);
		 return 1;
	 }
 
	 // Test unseen data
	 num_samples = 0;
	 correct = 0;
	 for (int i = 0; i < data->num_rows; i++) {
		 num_samples++;
		 prediction = nn_predict_quantized(network, data->input[i]);
		 true_positive = 0;
		 false_positive = 0;
 
		 for (int j = 0; j < output_size; j++) {
			 if (data->target[i][j] >= 0.5) {
				 if (prediction[j] >= 0.5)
					 true_positive++;
			 } else {
				 if (prediction[j] >= 0.5)
					 false_positive++;
			 }

			// free(prediction); 
			 
		 }
 
		 if ((true_positive == 1) && (false_positive == 0))
			 correct++;
	 }
	 printf("Test: %d/%d = %2.2f%%\n", correct, num_samples, (correct * 100.0) / num_samples);
	 //data_free(data);
 
	 // Cleanup
	 nn_free(network);  // Now frees both original and quantized data
	 return 0;
 }