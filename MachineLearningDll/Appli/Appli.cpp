
#include <iostream>
#include "LinearModel.h"
#include <vector>
#include <random> 

void printColoredPoint(double x, double y, int label) {
    
    if (label == 1) {
        std::cout << "\033[1;34m";  // Bleu
    }
    else {
        std::cout << "\033[1;31m";  // Rouge
    }

    std::cout << "(" << x << ", " << y << ")\033[0m" << std::endl;
}


int Cas2(){
    
    std::vector<std::vector<double>> X;
    std::vector<double> Y;
    srand(time(0));

    for (int i = 0; i < 50; ++i) {
        X.push_back({ static_cast<double>(rand()) / RAND_MAX * 0.9 + 1.0, static_cast<double>(rand()) / RAND_MAX * 0.9 + 1.0 });
        Y.push_back(1.0);
    }

    for (int i = 0; i < 50; ++i) {
        X.push_back({ static_cast<double>(rand()) / RAND_MAX * 0.9 + 2.0, static_cast<double>(rand()) / RAND_MAX * 0.9 + 2.0 });
        Y.push_back(-1.0);
    }
	
    std::cout << "Points X en couleur : " << std::endl;
    for (size_t i = 0; i < X.size(); ++i) {
        printColoredPoint(X[i][0], X[i][1], static_cast<int>(Y[i]));
    }

    LinearModel* model = Init();
    double learning_rate = 0.001;
    int epoch = 100000;
    Entrainement(model, X, Y, learning_rate, epoch);

    std::vector<std::vector<double>> X_test = {
       {2.0,2.0},
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
	

int Cas1() {
	
    std::vector<std::vector<double>> X_train = {
        {1, 1},
        {2, 3},
        {3, 3}
    };

    std::vector<double> y_train = { 1, -1, -1 };

 
    std::cout << "Points X_train en couleur : " << std::endl;
    for (size_t i = 0; i < X_train.size(); ++i) {
        printColoredPoint(X_train[i][0], X_train[i][1], static_cast<int>(y_train[i]));
    }

   
    LinearModel* model = Init();
    double learning_rate = 0.01;
    int epoch = 10000;
    Entrainement(model, X_train, y_train, learning_rate, epoch);

    std::vector<std::vector<double>> X_test = {
       {1.0, 1.0},
       {2.6, 3.0},
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
int Cas3() {

    std::vector<std::vector<double>> X = {
       {1, 0},
       {0, 1},
       {0, 0},
       {1, 1}
    };

    std::vector<double> Y = { 1, 1, -1, -1 };

    std::cout << "Points X en couleur : " << std::endl;
    for (size_t i = 0; i < X.size(); ++i) {
        printColoredPoint(X[i][0], X[i][1], static_cast<int>(Y[i]));

        LinearModel* model = Init();
        double learning_rate = 0.1;
        int epoch = 10000;
        Entrainement(model, X, Y, learning_rate, epoch);

        std::vector<std::vector<double>> X_test = {
            {1.0, 0.0},
            {0.0, 1.0},
            {0.0, 0.0},
            {1.0, 1.0}
        };

        std::vector<double> predictions;
        Prediction(model, X_test, predictions);


        std::cout << "Predictions en couleur :" << std::endl;
        for (size_t i = 0; i < X_test.size(); ++i) {
            printColoredPoint(X_test[i][0], X_test[i][1], static_cast<int>(predictions[i]));
        }


        Detruire(model);

        return 0;
    }
}




int main() {
    //Cas1();
    //Cas2();
    Cas3();
	return 0;
}