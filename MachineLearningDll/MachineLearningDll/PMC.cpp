#include "pch.h"
#include "PMC.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <cassert>
#include <algorithm> 
#include <functional> 

PMC::PMC(const std::vector<int>& npl) : layer_sizes(npl) {
    // Initialisation des poids et des autres membres
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (size_t i = 1; i < npl.size(); ++i) {
        std::vector<std::vector<double>> layer_weights;
        for (int j = 0; j < npl[i]; ++j) {
            std::vector<double> neuron_weights;
            for (int k = 0; k <= npl[i - 1]; ++k) {
                neuron_weights.push_back(dis(gen));
            }
            layer_weights.push_back(neuron_weights);
        }
        weights.push_back(layer_weights);
        activations.push_back(std::vector<double>(npl[i] + 1, 1.0));
        deltas.push_back(std::vector<double>(npl[i] + 1, 0.0));
    }
}

double PMC::activation_function(double x) {
    // Code de la fonction d'activation (par exemple, la tangente hyperbolique)
    return std::tanh(x);
}

double PMC::activation_derivative(double x) {
    // Code de la dérivée de la fonction d'activation
    return 1.0 - x * x;
}

void PMC::forward_propagate(const std::vector<double>& input) {
    // Code de la propagation avant
    assert(input.size() == layer_sizes[0]);

    activations[0] = std::vector<double>(input.begin(), input.end());
    activations[0].insert(activations[0].begin(), 1.0);

    for (size_t i = 1; i < layer_sizes.size(); ++i) {
        for (size_t j = 0; j < layer_sizes[i]; ++j) {
            double activation = 0.0;
            for (size_t k = 0; k <= layer_sizes[i - 1]; ++k) {
                activation += weights[i - 1][j][k] * activations[i - 1][k];
            }
            activations[i][j + 1] = activation_function(activation);
        }
    }
}

std::vector<double> PMC::predict(const std::vector<double>& input) {
    // Code de la prédiction
    forward_propagate(input);
    return std::vector<double>(activations.back().begin() + 1, activations.back().end());
}

void PMC::train(const std::vector<std::vector<double>>& inputs,
    const std::vector<double>& expected_outputs,
    double alpha, int max_iter) {
    // Code de l'entraînement
    for (int iter = 0; iter < max_iter; ++iter) {
        double total_error = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            forward_propagate(inputs[i]);

            for (size_t j = 0; j < layer_sizes.back(); ++j) {
                double error = expected_outputs[i] - activations.back()[j + 1];
                deltas.back()[j + 1] = error * activation_derivative(activations.back()[j + 1]);
                total_error += error * error;
            }

            for (int l = layer_sizes.size() - 2; l >= 0; --l) {
                for (size_t j = 0; j < layer_sizes[l]; ++j) {
                    double error = 0.0;
                    for (size_t k = 0; k < layer_sizes[l + 1]; ++k) {
                        error += deltas[l + 1][k] * weights[l][k][j];
                    }
                    deltas[l][j] = error * activation_derivative(activations[l][j]);
                }
            }

            const double SOME_THRESHOLD = 0.001;
            for (size_t l = 0; l < weights.size(); ++l) {
                for (size_t j = 0; j < weights[l].size(); ++j) {
                    for (size_t k = 0; k < weights[l][j].size(); ++k) {
                        weights[l][j][k] += alpha * deltas[l + 1][j] * activations[l][k];
                        if (total_error / inputs.size() < SOME_THRESHOLD) {
                            break;
                        }
                    }
                }
            }
            std::cout << "Epoch " << iter << " - Average Error: " << total_error / inputs.size() << std::endl;
            if (total_error / inputs.size() < SOME_THRESHOLD) {
                break;
            }
        }
    }
}
