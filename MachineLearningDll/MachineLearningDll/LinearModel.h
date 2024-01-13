//Created by Elodie
#ifdef LINEARMODEL_EXPORTS
#define LINEARMODEL_API __declspec(dllexport)
#else
#define LINEARMODEL_API __declspec(dllimport)
#endif


class LinearModel {
public:
    LINEARMODEL_API void Train(const double* X_train, int num_samples, int num_features, const double* y_train, double learning_rate, int epoch);

    LINEARMODEL_API void Predict(const double* X_test, int num_samples, int num_features, double* predictions);

private:
    double* poids;
    double bias;
};

extern "C" LINEARMODEL_API LinearModel * Init();

extern "C" LINEARMODEL_API void Detruire(LinearModel * model);

extern "C" LINEARMODEL_API void Entrainement(LinearModel * model, const double* X_train, int num_samples, int num_features, const double* y_train, double learning_rate, int epoch);

extern "C" LINEARMODEL_API void Prediction(LinearModel * model, const double* X_test, int num_samples, int num_features, double* predictions);