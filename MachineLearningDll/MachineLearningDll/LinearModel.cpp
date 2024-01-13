//Created by Elodie
#include "pch.h"
#include "LinearModel.h"
#include <iostream>


using namespace std; 

void LinearModel::Train(const double* X_train, int num_samples, int num_features, const double* y_train, double learning_rate, int epoch) {
    poids = new double[num_features];
    bias = 0.0;

    for (int i = 0; i < epoch; ++i) {
        double* predictions = new double[num_samples];

       
        for (int echantillon = 0; echantillon < num_samples; ++echantillon) {
            double prediction = bias;
            for (int j = 0; j < num_features; ++j) {
                prediction += poids[j] * X_train[echantillon * num_features + j];
            }
            predictions[echantillon] = prediction;
        }

        double* erreurs = new double[num_samples];
        for (int j = 0; j < num_samples; ++j) {
            erreurs[j] = predictions[j] - y_train[j];
        }

     
        for (int j = 0; j < num_features; ++j) {
            double gradient = 0.0;
            for (int k = 0; k < num_samples; ++k) {
                gradient += erreurs[k] * X_train[k * num_features + j];
            }
            poids[j] -= learning_rate * gradient;
        }

    
        double bias_gradient = 0.0;
        for (int j = 0; j < num_samples; ++j) {
            bias_gradient += erreurs[j];
        }
        bias -= learning_rate * bias_gradient;

      
        delete[] predictions;
        delete[] erreurs;
    }
}

void LinearModel::Predict(const double* X_test, int num_samples, int num_features, double* predictions) {
    for (int echantillon = 0; echantillon < num_samples; ++echantillon) {
        double prediction = bias;
        for (int i = 0; i < num_features; ++i) {
            prediction += poids[i] * X_test[echantillon * num_features + i];
        }
        predictions[echantillon] = prediction;
    }
}

LinearModel* Init() {
    return new LinearModel();
}

void Detruire(LinearModel* model) {
    delete model;
}

void Entrainement(LinearModel* model, const double* X_train, int num_samples, int num_features, const double* y_train, double learning_rate, int epoch) {
    model->Train(X_train, num_samples, num_features, y_train, learning_rate, epoch);
}

void Prediction(LinearModel* model, const double* X_test, int num_samples, int num_features, double* predictions) {
    model->Predict(X_test, num_samples, num_features, predictions);
}