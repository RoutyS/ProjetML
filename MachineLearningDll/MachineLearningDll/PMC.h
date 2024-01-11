// PMC.h
#ifndef PMC_H
#define PMC_H

#include <vector>

class PMC {
private:
    std::vector<int> layer_sizes;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> deltas;

public:
    PMC(const std::vector<int>& npl);
    double activation_function(double x);
    double activation_derivative(double x);
    void forward_propagate(const std::vector<double>& input);
    std::vector<double> predict(const std::vector<double>& input);
    void train(const std::vector<std::vector<double>>& inputs,
        const std::vector<double>& expected_outputs,
        double alpha, int max_iter);
};

#endif // PMC_H