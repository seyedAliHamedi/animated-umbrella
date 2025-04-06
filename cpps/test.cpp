#include <iostream>
#include <vector>
#include <numeric>
#include <bits/algorithmfwd.h>

// A simple function to calculate sum of a vector
double sum_vector(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0);
}

// A function to calculate average
double average(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return sum_vector(values) / values.size();
}

// A class for statistical operations
class Statistics {
private:
    std::vector<double> data;

public:
    Statistics(std::vector<double> input_data) : data(input_data) {}
    
    double mean() const {
        return average(data);
    }
    
    double max() const {
        if (data.empty()) return 0.0;
        return *std::max_element(data.begin(), data.end());
    }
    
    double min() const {
        if (data.empty()) return 0.0;
        return *std::min_element(data.begin(), data.end());
    }
    
    void add_value(double value) {
        data.push_back(value);
    }
    
    void print_data() const {
        std::cout << "Data: ";
        for (double value : data) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
};