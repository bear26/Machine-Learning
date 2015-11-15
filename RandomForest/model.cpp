#include "model.h"

#include <algorithm>
#include <iostream>
#include <cmath>
#include <numeric>

const int count_class = 2;

Model::Model()
{

}

void Model::train(const Data &data)
{
    const int num_trees = 23;
    forest_.resize(num_trees);

#pragma omp parallel for
    for(int i = 0; i < num_trees; ++i)
    {
        std::vector<int> permutation(data[0].features().size());
        std::iota(permutation.begin(), permutation.end(), 0);

        std::random_shuffle(permutation.begin(), permutation.end());

        Data sub_data;

        std::vector<int> perm(data.size());
        std::iota(perm.begin(), perm.end(), 0);

        std::random_shuffle(perm.begin(), perm.end());

        for(int j = 0; j < 40; ++j)
        {
            sub_data.add(data[perm[j]]);
        }

        forest_[i] = Tree::train(sub_data, std::vector<int>(permutation.begin(), permutation.begin() + 100));
    }
}

int Model::predict(const Object &object) const
{
    std::vector<int> ans(2);

    for(const auto &tree : forest_)
    {
        ++ans[tree->predict(object)];
    }

    return (ans[0] > ans[1]) ? 0 : 1;
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
