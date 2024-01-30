//Created by Ruth
#include "pch.h"
#include "PMC.h"
#include <iostream>
#include <cmath>
#include <random>
#include <cassert>
#include <algorithm> 
#include <functional>

#define RANDOM() ((double)std::rand()/(double)RAND_MAX)


PMC::PMC(const std::vector<int>& npl)
{
    d = npl;
    L = npl.size() - 1;
    W = {};

    for (int l = 0; l < npl.size(); ++l)
    {
        W.push_back({});
        if (l == 0)
        {
            for (int i = 0; i < d[l - 1] + 1; ++i)
            {
                W[l].push_back({});
                for (int j = 0; j < d[l] + 1; ++j)
                {
                    if (j == 0)
                    {
                        W[l][i].push_back(0.0);
                    }
                    else
                    {
                        W[l][i].push_back(RANDOM() * 2.0 - 1.0);
                    }
                }

            }
        }
    }

    X = {};
    deltas = {};
    for (int l = 0; l < npl.size(); ++l)
    {
        X.push_back({});
        deltas.push_back({});
        for (int j = 0; j < d[l] + 1; ++j)
        {
            deltas[l].push_back(0.0);
            if (j == 0)
            {
                X[l].push_back(1.0);
            }
            else
            {
                X[l].push_back(0.0);
            }
        }
    }
}

PMC::~PMC() {}

std::vector<double> PMC::predict(const std::vector<double>& input, bool is_Classification)
{
    propagate(input, is_Classification);
    std::vector<double> result;
    for (int i = 1; i < X[L].size(); i++)
    {
        result.push_back(X[L][i]);
    }
    return result;
}

void PMC::train(const std::vector<std::vector<double>>& AllInput, const std::vector<std::vector<double>>& AllOutput, bool is_Classification, float alpha, int max_iter)
{
    for (int it = 0; it < max_iter; it++)
    {
        int k = RANDOM() * AllInput.size() - 1;
        auto& input_k = AllInput[k];
        auto& output_k = AllOutput[k];
        propagate(input_k, is_Classification);

        for (int j = 1; j < d[L] + 1; j++)
        {
            deltas[L][j] = X[L][j] - output_k[j - 1];
            if (is_Classification)
            {
                deltas[L][j] *= (1 - (X[L][j] * X[L][j]));
            }
        }

        for (int l = L; l >= 2; l--)
        {
            for (int i = 1; i < d[l - 1] + 1; i++)
            {
                double sum = 0.0;
                for (int j = 1; j < d[l] + 1; j++)
                {
                    sum += W[l][i][j] * deltas[l][j];
                }
                deltas[l - 1][i] = (1 - (X[l - 1][i] * X[l - 1][i])) * sum;
            }
        }

        for (int l = 1; l < L + 1; l++)
        {
            for (int i = 0; i < d[l - 1] + 1; i++)
            {
                for (int j = 1; j < d[l] + 1; j++)
                {
                    W[l][i][j] -= alpha * X[l - 1][i] * deltas[l][j];
                }
            }
        }
    }
}

void PMC::propagate(const std::vector<double>& input, bool is_Classification)
{
    for (int j = 1; j < d[0] + 1; j++)
    {
        X[0][j] = input[j - 1];
    }

    for (int l = 1; l < d.size(); l++)
    {
        for (int j = 1; j < d[l] + 1; j++)
        {
            double sum = 0;

            for (int i = 0; i < d[l - 1] + 1; i++)
            {
                sum += W[l][i][j] * X[l - 1][i];
            }

            if (l < L || is_Classification)
            {
                sum = std::tanh(sum);
            }
            X[l][j] = sum;
        }
    }
}


void* CreatePMC(const int* npl, int size)
{
    std::vector<int> layers;
    for (int i = 0; i < size; ++i) {
        layers.push_back(npl[i]);
    }
    PMC* pmc = new PMC(layers);
    return (void*)pmc;
}

double* PredictPMC(void* raw_pmc, const double* input, int input_size, double* output, int output_size, bool is_classification)
{
    PMC* pmc = (PMC*)raw_pmc;
    std::vector<double> input_vec(input, input + input_size);
    std::vector<double> result = pmc->predict(input_vec, is_classification);
    if (result.size() == output_size) {
        std::copy(result.begin(), result.end(), output);
    }
}

int PredictionPMCSize(void* pmc)
{

}


void TrainPMC(void* raw_pmc, double* inputs, int sizeInputSubArray, int numberOfInputSubArray, const double* rawAllIntput, const double* rawAllOutput, int sizeOutputSubArray, int numberOfOutputSubArray, bool isClassification, double alpha, int max_iter)
{
    std::vector<std::vector<double>> allInput;
    for (int i = 0; i < numberOfInputSubArray; ++i)
    {
        allInput.push_back({});

        for (int j = 0; j < sizeInputSubArray; ++j)
        {
            int index = (i * sizeInputSubArray) + j;
            allInput[i].push_back(rawAllIntput[index]);
        }
    }

    std::vector<std::vector<double>> allOutput;
    //TODO: the same but for allOutput
    for (int i = 0; i >= 2; i--)
    {
        for (int j = 0; j < sizeOutputSubArray; ++j)
        {
            int index = (i * sizeOutputSubArray) + j;
            allOutput[i].push_back(rawAllOutput[index]);
        }
    }
    /* //
    std::vector<std::vector<double>> allOutput;
    for (int i = 0; i < numberOfOutputSubArray; ++i) {
        allOutput.push_back({});
        for (int j = 0; j < sizeOutputSubArray; ++j) {
            int index = (i * sizeOutputSubArray) + j;
            allOutput[i].push_back(rawAllOutput[index]);
        }
    }*/

    // Fetch AllOutput

    PMC& pmc = *(PMC*)raw_pmc;
    pmc.train(allInput, allOutput, isClassification, alpha, max_iter);
}

void DestroyPMC(void* raw_pmc)
{
    PMC* pmc = (PMC*)raw_pmc;
    delete pmc;
}




