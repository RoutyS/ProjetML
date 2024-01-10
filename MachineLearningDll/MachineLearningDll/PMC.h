#pragma once
class PMC
{
};

#ifndef PMC_HPP
#define PMC_HPP

#include <vector>

class PMC
{
public:
    PMC(const std::vector<int>& npl);
    void train(const std::vector<std::vector<double>>& inputs, const std::vector<double>& expected_outputs, double alpha, int max_iter);
    std::vector<double> predict(const std::vector<double>& input);

private:
    std::vector<int> layer_sizes;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> deltas;

    double activation_function(double x);
    double activation_derivative(double x);
    void forward_propagate(const std::vector<double>& input);
};

#endif // PMC_HPP
