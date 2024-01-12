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


int Cas1PMC() {
    // Création d'une PMC avec une architecture spécifique (à remplacer par vos valeurs)
    int layer_sizes[] = { 2, 4, 1 }; // Exemple d'architecture avec 2 neurones en entrée, 4 dans une couche cachée et 1 en sortie
    int num_layers = 3; // Nombre total de couches

    void* pmc = CreatePMC(layer_sizes, num_layers);

    // Entrées d'apprentissage
    double X_train[] = { 1, 1, 2, 3, 3, 3 }; // Exemple de données d'entrée
    int input_width = 2; // Nombre de caractéristiques par échantillon (dans cet exemple, 2D)
    int input_height = 3; // Nombre total d'échantillons

    // Sorties attendues
    double y_train[] = { 1, -1, -1 }; // Exemple de sorties attendues

    // Entraînement de la PMC avec les données fournies
    double learning_rate = 0.01;
    int max_iter = 10000;
    TrainPMC(pmc, X_train, input_width, input_height, y_train, 1, learning_rate, max_iter);

    // Prédictions avec la PMC
    double X_test[] = { 1.0, 1.0, 2.6, 3.0, 3.0, 3.0 }; // Exemple de données de test
    int input_size = 2; // Nombre de caractéristiques par échantillon dans les données de test

    double* raw_predictions = PredictPMC(pmc, X_test, input_size);

    // Affichage des prédictions (à adapter en fonction de votre application)
    std::cout << "Predictions:" << std::endl;
    for (int i = 0; i < input_height; ++i) {
        std::cout << raw_predictions[i] << std::endl;
    }

    // Libérer la mémoire de la PMC
    DestroyPMC(pmc);

    return 0;
}

int main() {
    //Cas1();
    //Cas2();
    //Cas3();
    Cas1PMC();

    return 0;
}