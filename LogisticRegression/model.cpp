#include "model.h"

#include <algorithm>
#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>

const int count_class = 2;

Model::Model()
{

}

void Model::train(const Data &data)
{
    w_.resize(data[0].features().size());

    std::generate(w_.begin(), w_.end(), []() { return 2 * double(rand()) / RAND_MAX - 1.0; });

    cv::Mat_<double> f(data.size(), data[0].features().size());
    cv::Mat_<double> y(data.size(), 1);

    for(size_t i = 0; i < data.size(); ++i)
    {
        for(size_t j = 0; j < data[i].features().size(); ++j)
        {
            f(i, j) = data[i].features()[j];
        }

        y(i, 0) = data[i].label();
    }

    while(true)
    {
        cv::Mat_<double> sigma(data.size(), 1);

        for(size_t i = 0; i < data.size(); ++i)
        {
            double sig = 0;
            for(size_t j = 0; j < data[i].features().size(); ++j)
            {
                sig += data[i].features()[j] * w_[j];
            }

            sig *= data[i].label();

            sig = 1 / (1 + std::exp(-sig));

            sigma(i, 0) = std::sqrt((1 - sig) * sig);
        }

        cv::Mat_<double> g = cv::Mat::diag(sigma);

        cv::Mat_<double> f_wave = g * f;

        cv::transpose(f_wave, f_wave);

        cv::Mat_<double> y_wave = y.mul(sigma);

        cv::Mat_<double> newton_direction = f_wave * y_wave;

        double diff = 0;

        for(size_t i = 0; i < w_.size(); ++i)
        {
            w_[i] += newton_direction(i, 0);

            diff += std::pow(newton_direction(i, 0), 2);
        }

        if (diff < 1e-4)
        {
            break;
        }

        //std::cout << diff << std::endl;
    }
}

int Model::predict(const Object &object) const
{
    std::vector<double> features = object.features();

    double cost = 0;

    for(size_t i = 0; i < features.size(); ++i)
    {
        cost += features[i] * w_[i];
    }

    return (cost < 0) ? -1 : 1;
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
