//created by Elodie
#include "pch.h"
#include "RBFN.h"
#include <iostream>

RBFN::RBFN() {
    for (int i = 0; i < num_centers; ++i) {
        centers[i] = i + 1.0;
        sigmas[i] = 1.0;
        weights[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);  
    }
}
float RBFN::rbf_approximation(float input) {
    float output = 0.0f;
    for (int i = 0; i < num_centers; ++i) {
        float term = std::exp(-(std::pow(input - centers[i], 2)) / (2 * std::pow(sigmas[i], 2)));
        std::cout << "Term " << i << ": " << term << "\n";
        output += weights[i] * term;
    }
    std::cout << "Output: " << output << "\n";
    return output;
}

// Regression
void RBFN::train(const float* inputs, const float* targets, int num_samples, float learning_rate, int epochs) {

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < num_samples; ++i) {
            float input = inputs[i];
            float target = targets[i];
            float prediction = rbf_approximation(input);


            float error = target - prediction;
            for (int j = 0; j < num_centers; ++j) {
                weights[j] += learning_rate * error * std::exp(-(std::pow(input - centers[j], 2)) / (2 * std::pow(sigmas[j], 2)));
            }
        }
    }
}

float rbf_approximation_instance(float input, RBFN* rbfInstance) {
    return rbfInstance->rbf_approximation(input);
}
void RBFN::train_classification(const float* inputs, const int* targets, int num_samples, float learning_rate, int epochs) {

    const int num_classes = 2;


    for (int j = 0; j < num_centers; ++j) {
        weights[j] = 0.001;
    }


    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Epoch " << epoch << "\n";

        for (int i = 0; i < num_samples; ++i) {
            float input = inputs[i];
            float target = static_cast<float>(targets[i]);
            float prediction = rbf_approximation(input);

            float error = target - prediction;
            std::cout << "Input: " << input << ", Target: " << target << ", Prediction: " << prediction << ", Error: " << error << "\n";


            for (int j = 0; j < num_centers; ++j) {
                weights[j] += learning_rate * error * std::exp(-(std::pow(input - centers[j], 2)) / (2 * std::pow(sigmas[j], 2)));
            }
        }
    }


}

int RBFN::predict_classification(float input) {
    float result = rbf_approximation(input);

    return (result >= 0.5) ? 1 : -1;  // Utiliser -1 pour représenter la classe négative
}

void train_classification_rbfn(RBFN* rbfInstance, const float* inputs, const int* targets, int num_samples, float learning_rate, int epochs) {
    rbfInstance->train_classification(inputs, targets, num_samples, learning_rate, epochs);
}
int predict_classification_rbfn(RBFN* rbfInstance, float input) {
    return  rbfInstance->predict_classification(input);
}
void train_rbfn(RBFN* rbfInstance, const float* inputs, const float* targets, int num_samples, float learning_rate, int epochs) {
    rbfInstance->train(inputs, targets, num_samples, learning_rate, epochs);
}
