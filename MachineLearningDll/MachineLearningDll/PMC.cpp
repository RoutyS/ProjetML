#include "pch.h"
#include "PMC.h"
#include <iostream>
#include <cmath>
#include <random>
#include <cassert>
#include <algorithm> 
#include <functional>

using namespace std;

PMC::PMC(const int* layer_sizes, int num_layers) {
    // ... impl�mentation de la construction, allocation de m�moire, etc.
    this->num_layers = num_layers;

    // Allocation de layer_sizes
    this->layer_sizes = new int[num_layers];
    for (int i = 0; i < num_layers; ++i) {
        this->layer_sizes[i] = layer_sizes[i];
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    size_of_predictions = layer_sizes[num_layers - 1];

    // Allocation de weights
    weights = new double** [num_layers - 1];
    for (int i = 0; i < num_layers - 1; ++i) {
        weights[i] = new double* [layer_sizes[i + 1]];
        for (int j = 0; j < layer_sizes[i + 1]; ++j) {
            weights[i][j] = new double[layer_sizes[i] + 1];
            // Initialisation des poids
            for (int k = 0; k <= layer_sizes[i]; ++k) {
                weights[i][j][k] = dis(gen); // Assigner une valeur al�atoire au poids
            }
        }
    }

    // Allocation et initialisation de activations et deltas
    activations = new double* [num_layers];
    deltas = new double* [num_layers];
    for (int i = 0; i < num_layers; ++i) {
        activations[i] = new double[layer_sizes[i] + 1];
        deltas[i] = new double[layer_sizes[i] + 1];
        // Initialisation des activations et deltas
        for (int j = 0; j <= layer_sizes[i]; ++j) {
            activations[i][j] = 1.0; // ou une autre valeur initiale
            deltas[i][j] = 0.0;
        }
    }
}

PMC::~PMC() {
    // ... impl�mentation de la destruction, lib�ration de m�moire, etc.
     // Lib�rer la m�moire allou�e pour layer_sizes
    delete[] layer_sizes;

    // Lib�rer la m�moire allou�e pour weights
    for (int i = 0; i < num_layers - 1; ++i) {
        for (int j = 0; j < layer_sizes[i + 1]; ++j) {
            delete[] weights[i][j];
        }
        delete[] weights[i];
    }
    delete[] weights;

    // Lib�rer la m�moire allou�e pour activations et deltas
    for (int i = 0; i < num_layers; ++i) {
        delete[] activations[i];
        delete[] deltas[i];
    }
    delete[] activations;
    delete[] deltas;
}

void PMC::train(double* inputs, int input_width, int input_height, const double* expected_outputs, int outputs_size, double alpha, int max_iter) {
    // Boucle sur le nombre d'it�rations
    for (int iter = 0; iter < max_iter; ++iter) {
        // Boucle sur chaque �chantillon d'entr�e
        for (int i = 0; i < input_height; ++i) {
            // Propagation avant
            forward_propagate(inputs + i * input_width);

            // Propagation arri�re et mise � jour des poids
            back_propagate(expected_outputs + i * outputs_size);
            update_weights(alpha);
        }
        // Eventuellement, imprimer la progression de l'entra�nement ici
    }
}

double* PMC::predict(const double* input, int input_size) {
    // Propagation avant avec l'entr�e donn�e
    forward_propagate(input);

    // Retourner les activations de la derni�re couche (sortie du r�seau)
    double* output = new double[layer_sizes[num_layers - 1]];
    for (int i = 0; i < layer_sizes[num_layers - 1]; ++i) {
        output[i] = activations[num_layers - 1][i];
    }
    return output;
}

double PMC::activation_function(double x) {
    // Fonction sigmoid comme exemple
    return 1.0 / (1.0 + exp(-x));
}

double PMC::activation_derivative(double x) {
    // D�riv�e de la fonction sigmoid
    return x * (1.0 - x);
}

void PMC::forward_propagate(const double* input) {
    // La premi�re couche d'activations est simplement l'entr�e
    for (int i = 0; i < layer_sizes[0]; ++i) {
        activations[0][i] = input[i];
    }

    // Boucle sur les couches cach�es et la couche de sortie
    for (int i = 1; i < num_layers; ++i) {
        for (int j = 0; j < layer_sizes[i]; ++j) {
            double net_input = 0.0;

            // Calculer la somme pond�r�e des entr�es avec les poids associ�s
            for (int k = 0; k <= layer_sizes[i - 1]; ++k) {
                net_input += weights[i - 1][j][k] * activations[i - 1][k];
            }

            // Appliquer la fonction d'activation � la somme pond�r�e
            activations[i][j] = activation_function(net_input);
        }
    }
}

void PMC::back_propagate(const double* expected_outputs) {
    // Calculer les deltas pour la couche de sortie
    for (int i = 0; i < layer_sizes[num_layers - 1]; ++i) {
        // Utiliser la d�riv�e de la fonction d'activation
        deltas[num_layers - 1][i] = activation_derivative(activations[num_layers - 1][i]) * (expected_outputs[i] - activations[num_layers - 1][i]);
    }

    // Propagation des deltas vers les couches cach�es
    for (int i = num_layers - 2; i > 0; --i) {
        for (int j = 0; j < layer_sizes[i]; ++j) {
            double error_sum = 0.0;

            // Calculer la somme pond�r�e des erreurs des couches suivantes
            for (int k = 0; k < layer_sizes[i + 1]; ++k) {
                error_sum += weights[i][k][j] * deltas[i + 1][k];
            }

            // Utiliser la d�riv�e de la fonction d'activation
            deltas[i][j] = activation_derivative(activations[i][j]) * error_sum;
        }
    }
}

void PMC::update_weights(double alpha) {
    // Mise � jour des poids pour chaque connexion entre les neurones
    for (int i = 0; i < num_layers - 1; ++i) {
        for (int j = 0; j < layer_sizes[i + 1]; ++j) {
            for (int k = 0; k <= layer_sizes[i]; ++k) {
                // Mettre � jour les poids en utilisant le taux d'apprentissage et les deltas
                weights[i][j][k] += alpha * deltas[i + 1][j] * activations[i][k];
            }
        }
    }
}

int PMC::getPredictionSize() const {
    // Retournez la taille des pr�dictions en utilisant la variable membre
    return size_of_predictions;
}

extern "C" {

    void* CreatePMC(const int* npl, int size) {
        return new PMC(npl, size);
    }

    void TrainPMC(void* pmc, double* inputs, int inputWidth, int inputHeight, const double* expected_outputs, int outputsSize, double alpha, int max_iter) {
        static_cast<PMC*>(pmc)->train(inputs, inputWidth, inputHeight, expected_outputs, outputsSize, alpha, max_iter);
    }

    int PredictionPMCSize(void* pmc) {
        // Appelez la fonction de la classe PMC pour obtenir la taille des pr�dictions
        int predictionSize = static_cast<PMC*>(pmc)->getPredictionSize();

        return predictionSize;
    }

    double* PredictPMC(void* pmc, const double* input, int inputSize) {
        return static_cast<PMC*>(pmc)->predict(input, inputSize);
    }

}

