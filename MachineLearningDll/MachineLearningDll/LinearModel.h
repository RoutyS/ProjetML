//Created by Elodie

// linear_model.h

#ifdef LINEAR_MODEL_EXPORTS
#define LINEAR_MODEL_API __declspec(dllexport)
#else
#define LINEAR_MODEL_API __declspec(dllimport)
#endif

class LinearModel {
private:
    int num_features;
    double* weights;

public:
    LinearModel(int num_features);
    ~LinearModel();
    void train_regression(const double* X, const double* y, int num_samples, int num_features, double learning_rate, int iterations);
    void train_classification(const double* X, const int* y, int num_samples, int num_features, double learning_rate, int iterations);
    double predict_regression(const double* X, int num_features);
    int predict_classification(const double* X, int num_features);
    void printDebugInfo(int iteration) const;
};

extern "C" {
    LINEAR_MODEL_API LinearModel* create_linear_model(int num_features);
    LINEAR_MODEL_API void train_regression(LinearModel* model, const double* X, const double* y, int num_samples, int num_features, double learning_rate, int iterations);
    LINEAR_MODEL_API void train_classification(LinearModel* model, const double* X, const int* y, int num_samples, int num_features, double learning_rate, int iterations);
    LINEAR_MODEL_API double predict_regression(LinearModel* model, const double* X, int num_features);
    LINEAR_MODEL_API int predict_classification(LinearModel* model, const double* X, int num_features);
    LINEAR_MODEL_API void destroy_linear_model(LinearModel* model);
}