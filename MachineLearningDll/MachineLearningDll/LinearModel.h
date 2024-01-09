#ifndef LINEARMODEL_H
#define LINEARMODEL_H

#include <vector>

#ifdef LINEARMODEL_EXPORTS
#define LINEARMODEL_API __declspec(dllexport)
#else
#define LINEARMODEL_API __declspec(dllimport)
#endif

class LinearModel {
public:
  
    LINEARMODEL_API void Train(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train);

 
    LINEARMODEL_API std::vector<double> Predict(const std::vector<std::vector<double>>& X_test);
};

extern "C" {
    LINEARMODEL_API LinearModel* Create();
    LINEARMODEL_API void Destroy(LinearModel* model);
    LINEARMODEL_API void Entrainement(LinearModel* model, const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train);
    LINEARMODEL_API void Prediction(LinearModel* model, const std::vector<std::vector<double>>& X_test, std::vector<double>& predictions);
}
#endif