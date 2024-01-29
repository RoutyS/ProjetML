#include "pch.h"
#include "LinearModel.h"

#include <iostream>
#include <vector>

//Created by Elodie
LinearModel::LinearModel(int num_features) : num_features(num_features) {
    weights = new double[num_features]();
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < num_features; ++i) {
        weights[i] = (rand() % 1000) / 1000.0;  
    }
}

LinearModel::~LinearModel() {
    delete[] weights;
}
void LinearModel::printDebugInfo(int iteration) const {
    std::cout << "Iteration " << iteration << ": Weights: ";
    for (int i = 0; i < num_features; ++i) {
        std::cout << weights[i] << " ";
    }
    std::cout << std::endl;
}



void LinearModel::train_regression(const double* X, const double* y, int num_samples, int num_features, double learning_rate, int iterations) {
   
    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 0; i < num_features; ++i) {
            double gradient = 0.0;
            for (int j = 0; j < num_samples; ++j) {
                double prediction = predict_regression(&X[j * num_features], num_features);
                double error = prediction - y[j];
                gradient += error * X[j * num_features + i];
            }

            weights[i] -= learning_rate * gradient / num_samples;
        }

        //debug
        if (iter % 100 == 0) {
            printDebugInfo(iter);
        }
    }
}

void LinearModel::train_classification(const double* X, const int* y, int num_samples, int num_features, double learning_rate, int iterations)
{
    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 0; i < num_features; ++i) {
            double gradient = 0.0;
            for (int j = 0; j < num_samples; ++j) {
                double prediction = predict_classification(&X[j * num_features], num_features);
                
                double error = 1.0 / (1.0 + exp(-prediction)) - y[j];
                gradient += error * X[j * num_features + i];
            }

          
            weights[i] -= learning_rate * gradient / num_samples;
        }

        //debug
        if (iter % 100 == 0) {
            printDebugInfo(iter);
        }
    }
}

double LinearModel::predict_regression(const double* X, int num_features) {

    double result = 0.0;
    for (int i = 0; i < num_features; ++i) {
        result += weights[i] * X[i];
    }

    return result;
}

int LinearModel::predict_classification(const double* X, int num_features)
{
    double prediction = 1.0 / (1.0 + exp(-predict_regression(X, num_features)));

    return (prediction >= 0.5) ? 1 : 0;
}

LinearModel* create_linear_model(int num_features)
{
    return new LinearModel(num_features);
}

void train_regression(LinearModel* model, const double* X, const double* y, int num_samples, int num_features, double learning_rate, int iterations)
{

    model->train_regression(X, y, num_samples, num_features, learning_rate, iterations);

}

void train_classification(LinearModel* model, const double* X, const int* y, int num_samples, int num_features, double learning_rate, int iterations)
{
    model->train_classification(X, y, num_samples, num_features, learning_rate, iterations);
}

double predict_regression(LinearModel* model, const double* X, int num_features)
{
    return model->predict_regression(X, num_features);
}

int predict_classification(LinearModel* model, const double* X, int num_features)
{
    return model->predict_classification(X, num_features);
}


void destroy_linear_model(LinearModel* model)
{
    delete model;
}