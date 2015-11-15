#include "tree.h"

#include <set>
#include <limits>
#include <cmath>
#include <iostream>

namespace
{
double get_gini(const Data &data)
{
    // 2 - count class
    std::vector<double> freq(2, 0);

    for(auto v : data)
    {
        ++freq[v.label()];
    }

    double score = 1;

    for(auto f : freq)
    {
        score -= std::pow(f / data.size(), 2);
    }

    return score;
}
}

Tree::Tree()
    :num_split_features_(-1), label_(-1)
{

}

std::shared_ptr<Tree> Tree::train(const Data &data, const std::vector<int> &mask)
{
    std::shared_ptr<Tree> tree = std::make_shared<Tree>();

    tree->train_(data, mask);

    return tree;
}

int Tree::predict(const Object &obj) const
{
    if (label_ == -1)
    {
        if (concept_true_(obj))
        {
            return left_->predict(obj);
        }
        else
        {
            return right_->predict(obj);
        }
    }

    return label_;
}

void Tree::train_(const Data &data, const std::vector<int> &mask)
{
    std::vector<double> freq(2);

    for(auto v : data)
    {
        ++freq[v.label()];
    }

    for(size_t i = 0; i < freq.size(); ++i)
    {
        freq[i] /= data.size();

        if (freq[i] >= 0.9)
        {
            label_ = i;
            return ;
        }
    }

    if (data.size() <= 3)
    {
        if (freq[0] > freq[1])
        {
            label_ = 0;
        }
        else
        {
            label_ = 1;
        }

        return ;
    }
    get_best_split_(data, mask);

    Data left, right;

    split_(data, left, right);

    left_ = Tree::train(left, mask);
    right_ = Tree::train(right, mask);
}

void Tree::get_best_split_(const Data &data, const std::vector<int> &mask)
{
    double best_score = std::numeric_limits<double>::max();

    int best_features_split = -1;
    double best_seporator = 0;

    for(auto f : mask)
    {
        num_split_features_ = f;

        std::set<double> all_values;

        for(auto v : data)
        {
            all_values.insert(v.features()[f]);
        }

        double prev = std::numeric_limits<double>::max();

        for(auto v : all_values)
        {
            if (prev > v)
            {
                prev = v;
                continue;
            }

            if (rand() % 3 != 0)
            {
                continue;
            }

            seporator_ = (prev + v) / 2;

            Data left, right;

            split_(data, left, right);

            double gini = (get_gini(left) * left.size() / data.size() + get_gini(right) * right.size() / data.size());

            if (gini < best_score)
            {
                best_score = gini;

                best_features_split = f;
                best_seporator = seporator_;
            }
        }
    }

    seporator_ = best_seporator;
    num_split_features_ = best_features_split;
}

bool Tree::concept_true_(const Object &obj) const
{
    return obj.features()[num_split_features_] > seporator_;
}

void Tree::split_(const Data &data, Data &left, Data &right) const
{
    left = Data();
    right = Data();

    for(auto &obj : data)
    {
        if (concept_true_(obj))
        {
            left.add(obj);
        }
        else
        {
            right.add(obj);
        }
    }
}
