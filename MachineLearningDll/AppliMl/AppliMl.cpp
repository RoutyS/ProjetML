#include <iostream>
#include "LinearModel.h"
#include "PMC.h"
#include <vector>
#include <random> 
//Created by Elodie
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
//Created by Elodie
int Cas2() {
    double X_data[100][2];
    double Y_data[100];
    srand(time(0));

    for (int i = 0; i < 50; ++i) {
        X_data[i][0] = static_cast<double>(rand()) / RAND_MAX * 0.9 + 1.0;
        X_data[i][1] = static_cast<double>(rand()) / RAND_MAX * 0.9 + 1.0;
        Y_data[i] = 1.0;
    }

    for (int i = 50; i < 100; ++i) {
        X_data[i][0] = static_cast<double>(rand()) / RAND_MAX * 0.9 + 2.0;
        X_data[i][1] = static_cast<double>(rand()) / RAND_MAX * 0.9 + 2.0;
        Y_data[i] = -1.0;
    }

    std::cout << "Points X en couleur : " << std::endl;
    for (size_t i = 0; i < 100; ++i) {
        printColoredPoint(X_data[i][0], X_data[i][1], static_cast<int>(Y_data[i]));
    }

    LinearModel* model = Init();
    double learning_rate = 0.001;
    int epoch = 100000;
    Entrainement(model, &X_data[0][0], 100, 2, Y_data, learning_rate, epoch);

    double X_test_data[3][2] = {
        {2.0, 2.0},
        {2.0, 3.0},
        {3.0, 3.0}
    };

    double predictions[3];
    Prediction(model, &X_test_data[0][0], 3, 2, predictions);

    std::cout << "Predictions:" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        std::cout << predictions[i] << std::endl;
    }

    Detruire(model);

    return 0;
}
//Created by Elodie
int Cas1() {
    double X_train_data[3][2] = {
        {1, 1},
        {2, 3},
        {3, 3}
    };

    double y_train_data[3] = { 1, -1, -1 };

    std::cout << "Points X_train en couleur : " << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        printColoredPoint(X_train_data[i][0], X_train_data[i][1], static_cast<int>(y_train_data[i]));
    }

    LinearModel* model = Init();
    double learning_rate = 0.01;
    int epoch = 100000;
    Entrainement(model, &X_train_data[0][0], 3, 2, y_train_data, learning_rate, epoch);

    double X_test_data[3][2] = {
       {1.0, 1.0},
       {2.6, 3.0},
       {3.0, 3.0}
    };

    double predictions[3];
    Prediction(model, &X_test_data[0][0], 3, 2, predictions);

    std::cout << "Predictions:" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        std::cout << predictions[i] << std::endl;
    }

    Detruire(model);

    return 0;
}
//Created by Elodie
int Cas3() {
    double X_data[4][2] = {
            {1, 0},
            {0, 1},
            {0, 0},
            {1, 1}
    };

    double Y_data[4] = { 1, 1, -1, -1 };

    std::cout << "Points X en couleur : " << std::endl;
    for (size_t i = 0; i < 4; ++i) {
        printColoredPoint(X_data[i][0], X_data[i][1], static_cast<int>(Y_data[i]));

        LinearModel* model = Init();
        double learning_rate = 0.1;
        int epoch = 10000;
        Entrainement(model, &X_data[0][0], 4, 2, Y_data, learning_rate, epoch);

        double X_test_data[4][2] = {
            {1.0, 0.0},
            {0.0, 1.0},
            {0.0, 0.0},
            {1.0, 1.0}
        };

        double predictions[4];
        Prediction(model, &X_test_data[0][0], 4, 2, predictions);

        std::cout << "Predictions en couleur :" << std::endl;
        for (size_t i = 0; i < 4; ++i) {
            printColoredPoint(X_test_data[i][0], X_test_data[i][1], static_cast<int>(predictions[i]));
        }

        Detruire(model);
    }

    return 0;
}

//Created by Ruth
int Cas1PMC() {
   
    int layer_sizes[] = { 2, 4, 1 }; 
    int num_layers = 3; 

    void* pmc = CreatePMC(layer_sizes, num_layers);

    double X_train[] = { 1, 1, 2, 3, 3, 3 }; 
    int input_width = 2; 
    int input_height = 3; 

   
    double y_train[] = { 1, -1, -1 }; 

    double learning_rate = 0.01;
    int max_iter = 10000;
    TrainPMC(pmc, X_train, input_width, input_height, y_train, 1, learning_rate, max_iter);

   
    double X_test[] = { 1.0, 1.0, 2.6, 3.0, 3.0, 3.0 }; 
    int input_size = 2;

    double* raw_predictions = PredictPMC(pmc, X_test, input_size);

    std::cout << "Predictions:" << std::endl;
    for (int i = 0; i < input_height; ++i) {
        std::cout << raw_predictions[i] << std::endl;
    }
 
    DestroyPMC(pmc);

    return 0;
}

int main() {
    Cas1();
    //Cas2();
    //Cas3();
    //Cas1PMC();

    return 0;
}