#pragma once

#ifdef RBFN_DLL_EXPORTS
#define RBFN_DLL_API __declspec(dllexport)
#else
#define RBFN_DLL_API __declspec(dllimport)
#endif

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

class RBFN {
private:
    ;

public:

    RBFN_DLL_API RBFN();

    RBFN_DLL_API float rbf_approximation(float input);
    RBFN_DLL_API void train(const float* inputs, const float* targets, int num_samples, float learning_rate, int epochs);

    RBFN_DLL_API void train_classification(const float* inputs, const int* targets, int num_samples, float learning_rate, int epochs);

    RBFN_DLL_API int predict_classification(float input);

    const int num_centers = 5;
    float centers[5];
    float sigmas[5];
    float weights[5];
};

extern "C" {
    RBFN_DLL_API float rbf_approximation_instance(float input, RBFN* rbfInstance);
    RBFN_DLL_API void train_rbfn(RBFN* rbfInstance, const float* inputs, const float* targets, int num_samples, float learning_rate, int epochs);
    RBFN_DLL_API void train_classification_rbfn(RBFN* rbfInstance, const float* inputs, const int* targets, int num_samples, float learning_rate, int epochs);
    RBFN_DLL_API int predict_classification_rbfn(RBFN* rbfInstance, float input);
}