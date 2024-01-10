// ConsoleApplication1.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//
#include <iostream>
#include "LinearModel.h"
#include <vector>
#include <random> 


int main() {
    std::vector<std::vector<double>> X_train = {
        {1, 1},
        {2, 3},
        {3, 3}
    };

    std::cout << "Points X_train : " << std::endl;
    
    for (const auto& point : X_train) {
        std::cout << "(" << point[0] << ", " << point[1] << ")" << std::endl;
    }

    std::vector<double> y_train = { 1, -1, -1 };

    LinearModel* model = Init();
    Entrainement(model, X_train, y_train);

    std::vector<std::vector<double>> X_test = {
       {1.0, 1.0},
       {2.0, 3.0},
       {3.0, 3.0}
    };


    std::vector<double> predictions;
    Prediction(model, X_test, predictions);

    std::cout << "Predictions:" << std::endl;
    for (double pred : predictions) {
        std::cout << pred << std::endl;
    }

    Detruire(model);

    return 0;
}