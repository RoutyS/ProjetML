//Created by Ruth
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
 
    this->num_layers = num_layers;

   
    this->layer_sizes = new int[num_layers];
    for (int i = 0; i < num_layers; ++i) {
        this->layer_sizes[i] = layer_sizes[i];
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    size_of_predictions = layer_sizes[num_layers - 1];

    weights = new double** [num_layers - 1];
    for (int i = 0; i < num_layers - 1; ++i) {
        weights[i] = new double* [layer_sizes[i + 1]];
        for (int j = 0; j < layer_sizes[i + 1]; ++j) {
            weights[i][j] = new double[layer_sizes[i] + 1];
            // Initialisation des poids
            for (int k = 0; k <= layer_sizes[i]; ++k) {
                weights[i][j][k] = dis(gen);
            }
        }
    }

  
    activations = new double* [num_layers];
    deltas = new double* [num_layers];
    for (int i = 0; i < num_layers; ++i) {
        activations[i] = new double[layer_sizes[i] + 1];
        deltas[i] = new double[layer_sizes[i] + 1];
    
        for (int j = 0; j <= layer_sizes[i]; ++j) {
            activations[i][j] = 1.0; 
            deltas[i][j] = 0.0;
        }
    }
}

PMC::~PMC() {
    
    delete[] layer_sizes;
	
    for (int i = 0; i < num_layers - 1; ++i) {
        for (int j = 0; j < layer_sizes[i + 1]; ++j) {
            delete[] weights[i][j];
        }
        delete[] weights[i];
    }
    delete[] weights;

   
    for (int i = 0; i < num_layers; ++i) {
        delete[] activations[i];
        delete[] deltas[i];
    }
    delete[] activations;
    delete[] deltas;
}

void PMC::train(double* inputs, int input_width, int input_height, const double* expected_outputs, int outputs_size, double alpha, int max_iter) {
   
    for (int iter = 0; iter < max_iter; ++iter) {
      
        for (int i = 0; i < input_height; ++i) {
           
            forward_propagate(inputs + i * input_width);

         
            back_propagate(expected_outputs + i * outputs_size);
            update_weights(alpha);
        }
      
    }
}
void PMC::ImageProcessing(const char* lien_Image) {

    std::ifstream file(lien_Image, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Impossible de charger les images." << std::endl;
        return;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    char* buffer = new char[size];
    if (!file.read(buffer, size)) {
        std::cerr << "Impossible de lire les informations de l'image." << std::endl;
        delete[] buffer;
        return;
    }

    file.close();


    std::cout << "Tilla de l'image: " << size << " bytes" << std::endl;

}
double* PMC::predict(const double* input, int input_size) {
   
    forward_propagate(input);

 
    double* output = new double[layer_sizes[num_layers - 1]];
    for (int i = 0; i < layer_sizes[num_layers - 1]; ++i) {
        output[i] = activations[num_layers - 1][i];
    }
    return output;
}

double PMC::activation_function(double x) {
    
    return 1.0 / (1.0 + exp(-x));
}

double PMC::activation_derivative(double x) {
   
    return x * (1.0 - x);
}

void PMC::forward_propagate(const double* input) {
   
    for (int i = 0; i < layer_sizes[0]; ++i) {
        activations[0][i] = input[i];
    }

    
    for (int i = 1; i < num_layers; ++i) {
        for (int j = 0; j < layer_sizes[i]; ++j) {
            double net_input = 0.0;

          
            for (int k = 0; k <= layer_sizes[i - 1]; ++k) {
                net_input += weights[i - 1][j][k] * activations[i - 1][k];
            }

            activations[i][j] = activation_function(net_input);
        }
    }
}

void PMC::back_propagate(const double* expected_outputs) {
   
    for (int i = 0; i < layer_sizes[num_layers - 1]; ++i) {
    
        deltas[num_layers - 1][i] = activation_derivative(activations[num_layers - 1][i]) * (expected_outputs[i] - activations[num_layers - 1][i]);
    }

   
    for (int i = num_layers - 2; i > 0; --i) {
        for (int j = 0; j < layer_sizes[i]; ++j) {
            double error_sum = 0.0;

         
            for (int k = 0; k < layer_sizes[i + 1]; ++k) {
                error_sum += weights[i][k][j] * deltas[i + 1][k];
            }

         
            deltas[i][j] = activation_derivative(activations[i][j]) * error_sum;
        }
    }
}

void PMC::update_weights(double alpha) {
   
    for (int i = 0; i < num_layers - 1; ++i) {
        for (int j = 0; j < layer_sizes[i + 1]; ++j) {
            for (int k = 0; k <= layer_sizes[i]; ++k) {
              
                weights[i][j][k] += alpha * deltas[i + 1][j] * activations[i][k];
            }
        }
    }
}

int PMC::getPredictionSize() const {
   
    return size_of_predictions;
}

void PMC::destroy() {
    
    delete[] layer_sizes;


    for (int i = 0; i < num_layers - 1; ++i) {
        for (int j = 0; j < layer_sizes[i + 1]; ++j) {
            delete[] weights[i][j];
        }
        delete[] weights[i];
    }
    delete[] weights;

    for (int i = 0; i < num_layers; ++i) {
        delete[] activations[i];
        delete[] deltas[i];
    }
    delete[] activations;
    delete[] deltas;
}


extern "C" {

    void* CreatePMC(const int* npl, int size) {
        return new PMC(npl, size);
    }

    void TrainPMC(void* pmc, double* inputs, int inputWidth, int inputHeight, const double* expected_outputs, int outputsSize, double alpha, int max_iter) {
        static_cast<PMC*>(pmc)->train(inputs, inputWidth, inputHeight, expected_outputs, outputsSize, alpha, max_iter);
    }

    int PredictionPMCSize(void* pmc) {
        int predictionSize = static_cast<PMC*>(pmc)->getPredictionSize();

        return predictionSize;
    }

    double* PredictPMC(void* pmc, const double* input, int inputSize) {
        return static_cast<PMC*>(pmc)->predict(input, inputSize);
    }

    void DestroyPMC(void* pmc) {
        delete static_cast<PMC*>(pmc);
    }

}

