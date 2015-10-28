#include "object.h"

#include <algorithm>
#include <cmath>
#include <cassert>

Object::Object()
{

}

Object::Object(int label, const std::vector<double> features)
    :label_(label)
{
    assert(features.size() == 2);

    //add new features like x1 * x2, x1 ^ 2 ...
    for(int i = 0; i <= 2; ++i)
    {
        for(int j = 0; j <= 2; ++j)
        {
            features_.push_back(std::pow(features[0], i) * std::pow(features[1], j));
        }
    }

    features_.push_back(std::sqrt(std::pow(features[0], 2) + std::pow(features[1], 2)));
    features_.push_back(std::cos(features[0]));
    features_.push_back(std::sin(features[0]));
    features_.push_back(std::cos(features[1]));
    features_.push_back(std::sin(features[1]));
}

bool Object::operator <(const Object &object) const
{
    return label_ < object.label_;
}


