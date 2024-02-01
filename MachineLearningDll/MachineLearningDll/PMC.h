//Created by Ruth
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
#include <fstream>
#include <cstdlib>
#include <cstring>
#ifdef __cplusplus

extern "C" {
#endif

    PMC_API void* CreatePMC(const int* npl, int size);
    PMC_API void TrainPMC(void* raw_pmc, double* inputs, int sizeInputSubArray, int numberOfInputSubArray, const double* rawAllIntput, const double* rawAllOutput, int sizeOutputSubArray, int numberOfOutputSubArray, bool isClassification, double alpha, int max_iter);
    PMC_API int PredictionPMCSize(void* raw_pmc);
    double* PredictPMC(void* raw_pmc, const double* input, int input_size, double* output, int output_size, bool is_classification);
    PMC_API void DestroyPMC(void* raw_pmc);


#ifdef __cplusplus
}
#endif


class PMC
{
public:
    PMC(const std::vector<int>& npl);
    ~PMC();
    std::vector<double> predict(const std::vector<double>& input, bool is_Classification = true);
    void train(const std::vector<std::vector<double>>& AllInput, const std::vector<std::vector<double>>& AllOutput, bool is_Classification = true, float alpha = 0.01, int max_iter = 1000);

private:

    std::vector<int> d;
    std::vector<std::vector<std::vector<double>>> W;
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> deltas;
    int L;

    void propagate(const std::vector<double>& input, bool is_Classification);

};

#endif