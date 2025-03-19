#ifndef NN_H
#define NN_H

// NN API Version
#define NN_VERSION_MAJOR    0
#define NN_VERSION_MINOR    1
#define NN_VERSION_PATCH    1
#define NN_VERSION_BUILD    0

// Forward declarations
typedef struct nn nn_t;
typedef struct nn_quantized nn_quantized_t;

typedef enum activation_function_type {
    ACTIVATION_FUNCTION_TYPE_NONE = 0,
    ACTIVATION_FUNCTION_TYPE_IDENTITY,
    ACTIVATION_FUNCTION_TYPE_LINEAR,
    ACTIVATION_FUNCTION_TYPE_RELU,
    ACTIVATION_FUNCTION_TYPE_LEAKY_RELU,
    ACTIVATION_FUNCTION_TYPE_ELU,
    ACTIVATION_FUNCTION_TYPE_THRESHOLD,
    ACTIVATION_FUNCTION_TYPE_SIGMOID,
    ACTIVATION_FUNCTION_TYPE_SIGMOID_FAST,
    ACTIVATION_FUNCTION_TYPE_TANH,
    ACTIVATION_FUNCTION_TYPE_TANH_FAST
} activation_function_type_t;

typedef struct nn_quantized {
    uint32_t depth;             // Number of layers
    uint32_t *width;            // Neurons per layer
    int8_t*** quantized_weights;
    float** weight_scales;
    int8_t** quantized_biases;
    float* bias_scales;
} nn_quantized_t;

typedef struct nn {
    uint32_t depth;
    uint32_t *width;
    uint8_t *activation;
    float *bias;
    float **neuron;
    float **loss;
    float **preact;
    float ***weight;
    float ***weight_adj;
    nn_quantized_t* quantized_network;
} nn_t;

// Function declarations
nn_t *nn_init(void);
void nn_free(nn_t *nn);
void nn_free_quantized(nn_quantized_t* quantized_network);
int nn_add_layer(nn_t *nn, int width, int activation, float bias);
int nn_save(nn_t *nn, char *path);
nn_t *nn_load(char *path);
nn_t* nn_load_quantized(const char* path);  // Updated return type
float nn_train(nn_t *nn, float *inputs, float *targets, float rate);
float *nn_predict(nn_t *nn, float *inputs);
float *nn_predict_quantized(nn_t *network, float *input);  // Updated parameter type
float activate(float value, int activation_type);
uint32_t nn_version(void);

#endif /* NN_H */