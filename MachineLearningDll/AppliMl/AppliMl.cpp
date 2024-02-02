#include <iostream>
#include "LinearModel.h"
#include "PMC.h"
#include <vector>
#include <random> 
#include "RBFN.h"
#include <cmath>

int main() {
    // Known dataset for testing
    const int num_samples = 10;
    const float inputs[num_samples] = { /* Your input values here */ };
    const float targets[num_samples] = { /* Your target values here */ };
    const int classification_targets[num_samples] = { /* Your classification targets here */ };

    // RBFN instance
    RBFN rbfInstance;

    // regression
    rbfInstance.train(inputs, targets, num_samples, 0.0001, 1000);
 
    std::cout << "Trained Weights for Regression:\n";
    for (int j = 0; j < rbfInstance.num_centers; ++j) {
        std::cout << "Weight[" << j << "]: " << rbfInstance.weights[j] << "\n";
    }

    //classification
    rbfInstance.train_classification(inputs, classification_targets, num_samples, 0.001, 10000);


    std::cout << "Trained Weights for Classification:\n";
    for (int j = 0; j < rbfInstance.num_centers; ++j) {
        std::cout << "Weight[" << j << "]: " << rbfInstance.weights[j] << "\n";
    }


    for (int i = 0; i < num_samples; ++i) {
        float input = inputs[i];
        float regression_prediction = rbfInstance.rbf_approximation(input);
        int classification_prediction = rbfInstance.predict_classification(input);

        std::cout << "Input: " << input << ", Regression Prediction: " << regression_prediction << ", Classification Prediction: " << classification_prediction << "\n";
    }

    return 0;
}