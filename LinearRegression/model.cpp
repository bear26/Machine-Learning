#include "model.h"

#include <cassert>
#include <limits>
#include <cmath>
#include <iostream>

Model::Model()
{

}

void Model::train(const Data &data)
{
    assert(data.size() != 0);

    Data norm_data = compute_normalize_(data);

    // add free variable
    for(auto &obj : norm_data)
    {
        obj.features().insert(obj.features().begin(), 1);
    }

    // gradient descent
    w_.resize(norm_data[0].features().size());

    for(auto &w : w_)
    {
        w = double(rand()) / RAND_MAX;
    }

    const double alpha = 0.1;

    while(true)
    {
        double error = 0;

        for(const auto &obj : norm_data)
        {
            double h = 0;

            for(size_t j = 0; j < w_.size(); ++j)
            {
                h += w_[j] * obj.features()[j];
            }

            error += std::pow(h - obj.label(), 2);
        }

        error /= 2 * norm_data.size();

        std::cout << error << "  ";

        std::vector<double> new_w(w_.size());

        for(size_t k = 0; k < new_w.size(); ++k)
        {
            double step = 0;
            for(const auto &obj: norm_data)
            {
                double h = 0;

                for(size_t j = 0; j < w_.size(); ++j)
                {
                    h += w_[j] * obj.features()[j];
                }

                step += (h - obj.label()) * obj.features()[k];
            }

            new_w[k] = w_[k] - step * alpha / norm_data.size();
        }

        double difference = 0;

        for(size_t i = 0; i < w_.size(); ++i)
        {
            difference += std::fabs(w_[i] - new_w[i]);
        }

        std::cout << difference << std::endl;

        if (difference < 1e-9)
        {
            break;
        }


        w_ = new_w;
    }
}

double Model::predict(const Data &data) const
{
    double ans = 0;

    Data norm_data = normalize_(data);

    for(auto obj : norm_data)
    {
        std::vector<double> features = obj.features();

        double cost = w_[0];

        for(size_t i = 0; i < features.size(); ++i)
        {
            cost += features[i] * w_[i + 1];
        }

        ans += std::fabs(cost - obj.label());
    }

    return ans / data.size();
}

double Model::predict(const std::vector<double> &features) const
{
    std::vector<double> norm_features(features.size());

    for(size_t i = 0; i < features.size(); ++i)
    {
        norm_features[i] = (features[i] - min_[i]) / (max_[i] - min_[i]);
    }

    double cost = w_[0];

    for(size_t i = 0; i < norm_features.size(); ++i)
    {
        cost += norm_features[i] * w_[i + 1];
    }

    return cost;
}

Data Model::compute_normalize_(const Data &data)
{
    min_.clear();
    max_.clear();

    for(size_t i = 0; i < data[0].features().size(); ++i)
    {
        min_.push_back(std::numeric_limits<double>::max());
        max_.push_back(std::numeric_limits<double>::lowest());

        for(size_t j = 0; j < data.size(); ++j)
        {
            min_[i] = std::min(min_[i], data[j].features()[i]);
            max_[i] = std::max(max_[i], data[j].features()[i]);
        }
    }

    return normalize_(data);
}

Data Model::normalize_(const Data &data) const
{
    std::vector<Object> ans;

    for(auto obj : data)
    {
        std::vector<double> norm_features(obj.features().size());

        for(size_t j = 0; j < obj.features().size(); ++j)
        {
            norm_features[j] = (obj.features()[j] - min_[j]) / (max_[j] - min_[j]);
        }

        ans.push_back(Object(obj.label(), norm_features));
    }

    return ans;
}
