#include "model.h"

#include <algorithm>
#include <iostream>
#include <cmath>

#include <boost/multiprecision/cpp_dec_float.hpp>

const int count_class = 2;

Model::Model()
{

}

void Model::train(const Data &data)
{
    prob_.resize(count_class);

    std::vector<int> count_in_class(prob_.size());

    for(const Object obj : data)
    {
        for(auto v : obj.body())
        {
            if (prob_[obj.label()].find(v) == prob_[obj.label()].end())
            {
                prob_[obj.label()][v] = 0;
            }

            ++prob_[obj.label()][v];

            ++count_in_class[obj.label()];
        }

        for(auto v : obj.subject())
        {
            if (prob_[obj.label()].find(v) == prob_[obj.label()].end())
            {
                prob_[obj.label()][v] = 0;
            }

            // weight subject more than body
            prob_[obj.label()][v] += 5;

            count_in_class[obj.label()] += 5;
        }
    }

    for(size_t i = 0; i < prob_.size(); ++i)
    {
        for(auto it : prob_[i])
        {
            prob_[i][it.first] /= count_in_class[i];
        }
    }
}

int Model::predict(const Object &object) const
{
    std::vector<boost::multiprecision::cpp_dec_float_100> ans(count_class);

    // spam probability less than leggit message
    ans[0] = 0.3;
    ans[1] = 0.7;

    for(size_t i = 0; i < ans.size(); ++i)
    {
        for(auto v : object.body())
        {
            ans[i] *= (prob_[i].find(v) != prob_[i].end()) ? -std::log(prob_[i].at(v)) : 1;
        }

        for(auto v : object.subject())
        {
            ans[i] *= (prob_[i].find(v) != prob_[i].end()) ? -std::log(prob_[i].at(v)) : 1;
        }
    }

    return std::distance(ans.begin(), std::max_element(ans.begin(), ans.end()));
}

std::vector<std::pair<int, int> > Model::test(const Data &data) const
{
    std::vector<std::pair<int, int>> ans(data.size());

    for(size_t i = 0; i < data.size(); ++i)
    {
        ans[i] = (std::make_pair(data[i].label(), predict(data[i])));
    }

    return ans;
}

double cross_validation(const Data &data, int folder, int t)
{
    std::cout << "Cross validation..." << std::endl;

    double ans = 0;
    for(int i = 0; i < t; ++i)
    {
        std::vector<Data> data_split;
        data.split(folder, data_split);

        std::vector<std::pair<int, int>> result;

        for(size_t i = 0; i < data_split.size(); ++i)
        {
            Data data_train;

            for(size_t j = 0; j < data_split.size(); ++j)
            {
                if (j != i)
                {
                    data_train.add(data_split[j]);
                }
            }

            Model model;
            model.train(data_train);

            auto curr_result = model.test(data_split[i]);
            result.insert(result.end(), curr_result.begin(), curr_result.end());
        }

        ans += print_result(result);
    }

   return ans / t;
}

double print_result(const std::vector<std::pair<int, int> > &result)
{
    std::vector<std::vector<int>> conf_matrix(count_class, std::vector<int>(count_class, 0));

    for(auto pair : result)
    {
        conf_matrix[std::max(0, pair.first)][std::max(0, pair.second)]++;
    }

    for(size_t i = 0; i < conf_matrix.size(); ++i)
    {
        double presision = 0;
        double recall = 0;

        for(size_t j = 0; j < conf_matrix[i].size(); ++j)
        {
            presision += conf_matrix[i][j];
        }

        for(size_t j = 0; j < conf_matrix.size(); ++j)
        {
            recall += conf_matrix[j][i];
        }

        presision = conf_matrix[i][i] / presision;
        recall = conf_matrix[i][i] / recall;

        printf("Fscore %d class: %lf\n", (int)i, 2 * presision * recall / (presision + recall));
    }

    int correct = 0;
    for(size_t i = 0; i < conf_matrix.size(); ++i)
    {
        correct += conf_matrix[i][i];
    }


    printf("Accuracy: %lf%%(%d/%d)\n", 100.0 * correct / result.size(), correct, (int)result.size());

    return 100.0 * correct / result.size();
}
