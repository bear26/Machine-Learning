#include "object.h"

#include <algorithm>
#include <cmath>

const auto gauss_func = [](double x, double m, double s)
{
    return 1 / (s * std::sqrt(2 * M_PI)) * std::pow(M_E, -std::pow(x - m, 2) / (2 * std::pow(s, 2)));
};

Object::Object()
{

}

Object::Object(int label, const std::vector<double> features)
    :_label(label), _features(features)
{

}

double Object::distance(const Object &obj) const
{
    double sum = 0;
    for(size_t i = 0; i < _features.size(); ++i)
    {
        sum += std::pow(_features[i] - obj._features[i], 2);
    }

    //add magic(axis z)
    const double z = 4;

    double dist_to_center1 = std::sqrt(std::pow(_features[0] - 0.15, 2) + std::pow(_features[1] - 0.15, 2));
    double dist_to_center2 = std::sqrt(std::pow(obj._features[0] - 0.15, 2) + std::pow(obj._features[1] - 0.15, 2));

    sum += std::pow(z * gauss_func(dist_to_center1, 0, std::sqrt(0.2)) - z * gauss_func(dist_to_center2, 0, std::sqrt(0.2)), 2);

    return std::sqrt(sum);
}

double Object::metric(const Object &obj, double max_distance) const
{
    double sum = 0;
    for(size_t i = 0; i < _features.size(); ++i)
    {
        sum += std::pow(_features[i] - obj._features[i], 2);
    }

    double dist = std::sqrt(sum) / max_distance;

    return gauss_func(dist, 0, std::sqrt(0.4));
}

bool Object::operator <(const Object &object) const
{
    return _label < object._label;
}


