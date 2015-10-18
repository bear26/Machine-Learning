#include "model.h"

#include <algorithm>
#include <iostream>

const int count_class = 2;

Model::Model()
{

}

int Model::predict(const Object &object) const
{
    std::vector<std::pair<double, Object>> dist_label(_data.size());

    for(size_t i = 0; i < _data.size(); ++i)
    {
        dist_label[i] = std::make_pair(object.distance(_data[i]), _data[i]);
    }

    std::nth_element(dist_label.begin(), dist_label.begin() + _k, dist_label.end());

    std::pair<double, Object> kth_element = dist_label[_k];

    std::vector<double> weight(count_class, 0);

    for(const auto &v : dist_label)
    {
        if (v < kth_element)
        {
            weight[v.second.label()] += object.metric(v.second, kth_element.first);
        }
    }

    int max = 0;
    for(size_t i = 1; i < weight.size(); ++i)
    {
        if (weight[max] <= weight[i])
        {
            max = i;
        }
    }

    return max;
}

std::vector<std::pair<int, int> > Model::test(const Data &data) const
{
    std::vector<std::pair<int, int>> ans(data.size());

#pragma omp parallel for
    for(size_t i = 0; i < data.size(); ++i)
    {
        ans[i] = (std::make_pair(data[i].label(), predict(data[i])));
    }

    return ans;
}

void Model::train(const Data &data, int k)
{
    _data = data;
    _k = k;
}

double cross_validation(const Data &data, int k, int folder, int t)
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
            model.train(data_train, k);

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
        conf_matrix[pair.first][pair.second]++;
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
