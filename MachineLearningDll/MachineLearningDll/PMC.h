
#ifndef PMC_H
#define PMC_H

#ifdef PMC_EXPORTS
#define PMC_API __declspec(dllexport)
#else
#define PMC_API __declspec(dllimport)
#endif


#include <random>
#include <iostream>
#include <cmath>
#include <random>
#include <cassert>
#include <algorithm> 
#include <functional>

#ifdef __cplusplus
extern "C" {
#endif

    PMC_API void* CreatePMC(const int* layer_sizes, int num_layers);
    PMC_API void TrainPMC(void* pmc, double* inputs, int input_width, int input_height, const double* expected_outputs, int output_size, double alpha, int max_iter);
    PMC_API int PredictionPMCSize(void* pmc);
    PMC_API double* PredictPMC(void* pmc, const double* input, int input_size);

#ifdef __cplusplus
}
#endif



class PMC {
private:
    int* layer_sizes;
    int num_layers;
    double*** weights; 
    double** activations; 
    double** deltas;
    int size_of_predictions;

public:
    PMC(const int* layer_sizes, int num_layers);
    ~PMC();  // D�finissez un destructeur pour g�rer la m�moire
    void train(double* inputs, int input_width, int input_height, const double* expected_outputs, int output_size, double alpha, int max_iter);
    double* predict(const double* input, int input_size);
    int getPredictionSize() const;

private:
    double activation_function(double x);
    double activation_derivative(double x);
    void forward_propagate(const double* input);
    void back_propagate(const double* expected_outputs);
    void update_weights(double alpha);

};

#endif // PMC_H