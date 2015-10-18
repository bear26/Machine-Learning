#ifndef MODEL_H
#define MODEL_H

#include "data.h"

class Model
{
public:
    Model();

    void train(const Data &data);

    // return mean error
    double predict(const Data &data) const;

    double predict(const std::vector<double> &features) const;

private:
    // for normalization
    Data compute_normalize_(const Data &data);
    Data normalize_(const Data &features) const;

    std::vector<double> min_;
    std::vector<double> max_;

    std::vector<double> w_;
};

#endif // MODEL_H

