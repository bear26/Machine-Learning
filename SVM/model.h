#ifndef MODEL
#define MODEL

#include "data.h"

class Model
{
public:
    Model();

    void train(const Data &data, double c);

    int predict(const Object &object) const;

    //return vector pair(real label, prediction label)
    std::vector<std::pair<int, int>> test(const Data &data) const;

private:
    Data sv_;
    std::vector<double> lambda_;
    double w0_;
    double c_;

    double u_(const std::vector<double> &features) const;
    std::vector<double> error_cache_;

    bool processing_(size_t j);
    bool processing2_(size_t j, double e_j);
    bool optimization_(size_t i, size_t j);

    void shuffle_();
};

double print_result(const std::vector<std::pair<int, int>> &result);
double cross_validation(const Data &data, int folder, int t);


#endif // MODEL

