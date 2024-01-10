
#ifdef LINEARMODEL_EXPORTS
#define LINEARMODEL_API __declspec(dllexport)
#else
#define LINEARMODEL_API __declspec(dllimport)
#endif
#include <vector>
#include <random>
class LinearModel {
public:

    LINEARMODEL_API void Train(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train, double learning_rate, int epoch);


    std::vector<double> Predict(const std::vector<std::vector<double>>& X_test);


private:
    std::vector<double> poids;
    double bias; // Le biais du mod�le
};


extern "C" LINEARMODEL_API  LinearModel * Init();

extern "C"    LINEARMODEL_API void Detruire(LinearModel * model);

extern "C"   LINEARMODEL_API void Entrainement(LinearModel * model, const std::vector<std::vector<double>>&X_train, const std::vector<double>&y_train, double learning_rate, int epoch);

extern "C"    LINEARMODEL_API void Prediction(LinearModel * model, const std::vector<std::vector<double>>&X_test, std::vector<double>&predictions);
