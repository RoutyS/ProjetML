#include "pch.h"
#include "LinearModel.h"
#include <iostream>

using namespace std; 
void LinearModel::Train(const vector<vector<double>>& X_train, const vector<double>& y_train) {
  
    poids.assign(X_train[0].size(), 0.0); 
    bias = 0.0;
	
    double learning_rate = 0.01;
    int epoch = 1000;

    int nb_echantillons = static_cast<int>(X_train.size());
    int num_features = static_cast<int>(X_train[0].size());
	
    for (int i = 0; i < epoch; ++i) {
        vector<double> predictions;
		
        for (size_t echantillon = 0; echantillon < nb_echantillons; ++echantillon) {
            double prediction = bias;
            for (size_t i = 0; i < poids.size(); ++i) {
                prediction += poids[i] * X_train[echantillon][i];
            }
            predictions.push_back(prediction);
        }

        vector<double> erreurs;
        for (int i = 0; i < nb_echantillons; ++i) {
            erreurs.push_back(predictions[i] - y_train[i]);
        }

        for (int i = 0; i < num_features; ++i) {
            double gradient = 0.0;
            for (int j = 0; j < nb_echantillons; ++j) {
                gradient += erreurs[j] * X_train[j][i];
            }
            poids[i] -= learning_rate * gradient;
        }

        double bias_gradient = 0.0;
        for (int i = 0; i < nb_echantillons; ++i) {
            bias_gradient += erreurs[i];
        }
        bias -= learning_rate * bias_gradient;
    }
}

vector<double> LinearModel::Predict(const vector<vector<double>>& X_test) {
    vector<double> predictions;
    for (size_t echantillon = 0;echantillon < X_test.size();echantillon++) {
        double prediction = bias;
        for (size_t i = 0; i < poids.size(); ++i) {
            prediction += poids[i] * X_test[echantillon][i];
        }
        predictions.push_back(prediction);
    }
    return predictions;
}

LinearModel* Init() {
    return new LinearModel();
}

void Detruire(LinearModel* model) {
    delete model;
}

void Entrainement(LinearModel* model, const vector<vector<double>>& X_train, const vector<double>& y_train) {
   
    model->Train(X_train, y_train);
}


void Prediction(LinearModel* model, const vector<vector<double>>& X_test, vector<double>& predictions) {
    predictions = model->Predict(X_test);
}

