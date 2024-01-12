#include <iostream>
#include "LinearModel.h"
#include "PMC.h"
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

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(1.0, 2.0);

int Cas2() {

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



/*
int Cas1PMC() {
    // Données d'entraînement
    std::vector<std::vector<double>> X_train = {
        {1, 1},
        {2, 3},
        {3, 3}
    };
    std::vector<double> y_train = { 1, -1, -1 };

    // Affichage des points d'entraînement en couleur
    std::cout << "Points X_train en couleur :" << std::endl;
    for (size_t i = 0; i < X_train.size(); ++i) {
        // Vérification pour éviter tout accès hors limites
        if (i < y_train.size()) {
            printColoredPoint(X_train[i][0], X_train[i][1], y_train[i]);
        }
    }

    // Initialisation et configuration du PMC (Perceptron Multicouche)
    std::vector<int> layers = { 2, 2, 1 }; // Exemple de structure du PMC
    PMC mlp(layers);

    // Entraînement du PMC
    double alpha = 0.01; // Taux d'apprentissage
    int max_iter = 10000; // Nombre d'itérations
    mlp.train(X_train, y_train, alpha, max_iter);

    // Données de test et prédiction
    std::vector<std::vector<double>> X_test = {
        {1.0, 1.0},
        {2.0, 3.0},
        {3.0, 3.0}
    };
    std::cout << "Prédictions :" << std::endl;
    for (const auto& x : X_test) {
        std::vector<double> prediction = mlp.predict(x);
        std::cout << (prediction.front() > 0 ? 1 : -1) << std::endl;
    }

    return 0;
}*/
/*
int Cas1PMC() {
    // Données d'entraînement
    std::vector<std::vector<double>> X_train = {
        {1, 1},
        {2, 3},
        {3, 3}
    };
    std::vector<double> y_train = { 1, -1, -1 };

    // Vérifiez que le nombre de points d'entraînement correspond au nombre d'étiquettes
    assert(X_train.size() == y_train.size() && "Le nombre de points d'entraînement doit correspondre au nombre d'étiquettes.");

    // Initialisation et configuration du PMC
    std::vector<int> layers = { 2, 2, 1 }; // Assurez-vous que cela correspond à la structure de votre réseau
    PMC mlp(layers);

    // Assurez-vous que la taille de la couche d'entrée correspond au nombre de caractéristiques de chaque point d'entraînement
    assert(layers.front() == X_train.front().size() && "La taille de la couche d'entrée doit correspondre au nombre de caractéristiques d'entrée.");

    // Entraînement du PMC
    double alpha = 0.01; // Taux d'apprentissage
    int max_iter = 10000; // Nombre d'itérations
    mlp.train(X_train, y_train, alpha, max_iter);

    // Données de test et prédiction
    std::vector<std::vector<double>> X_test = {
        {1.0, 1.0},
        {2.0, 3.0},
        {3.0, 3.0}
    };

    // Vérifiez que chaque point de test a la même dimension que la couche d'entrée du PMC
    for (const auto& test_point : X_test) {
        assert(test_point.size() == layers.front() && "Chaque point de test doit avoir la même dimension que la couche d'entrée du PMC.");
    }

    std::cout << "Prédictions :" << std::endl;
    for (const auto& x : X_test) {
        std::vector<double> prediction = mlp.predict(x);
        std::cout << (prediction.front() > 0 ? 1 : -1) << std::endl;
    }

    return 0;
}*/


int main() {
    Cas1();
    //Cas2();
    //Cas3();
    //Cas1PMC();

    return 0;
}