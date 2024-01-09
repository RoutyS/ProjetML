#include "pch.h"
#include "LinearModel.h"
#include <iostream>


void LinearModel::Train(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train) {
    std::cout << "Entrainement..." << std::endl;
}

std::vector<double> LinearModel::Predict(const std::vector<std::vector<double>>& X_test) {
    std::cout << "Prediction..." << std::endl;
  
    return std::vector<double>(); 
}

LinearModel* Create() {
    return new LinearModel();
}

void Destroy(LinearModel* model) {
    delete model;
}

void Entrainement(LinearModel* model, const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train) {
    model->Train(X_train, y_train);
}

void Prediction(LinearModel* model, const std::vector<std::vector<double>>& X_test, std::vector<double>& predictions) {
    predictions = model->Predict(X_test);
}