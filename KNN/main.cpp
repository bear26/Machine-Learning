#include <iostream>
#include <fstream>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "data.h"
#include "model.h"

void show_histogram(const std::vector<std::pair<int, int>> &acc, const std::vector<std::pair<int, int>> &cross)
{
    int hist_h = 150;
    int hist_w = 200;
    cv::Mat hist(hist_h, hist_w + 100, CV_8UC3, cv::Scalar(0, 0, 0));

    int max = 0;
    for(size_t i = 1; i < acc.size(); ++i)
    {
        cv::line(hist, cv::Point(acc[i - 1].first * (hist_w / acc.size()), hist_h - acc[i - 1].second), cv::Point(acc[i].first * (hist_w / acc.size()), hist_h - acc[i].second), cv::Scalar(0, 255, 0), 1, 1, 0);

        if (acc[max].second < acc[i].second)
        {
            max = i;
        }
    }

    cv::putText(hist, std::to_string(acc[max].first) + ", " + std::to_string(acc[max].second) + "%", cv::Point(acc[max].first * (hist_w / acc.size()), hist_h - acc[max].second - 10), cv::FONT_ITALIC, 0.4, cv::Scalar(0, 255, 0), 1, 1);

    max = 0;
    for(size_t i = 1; i < cross.size(); ++i)
    {
        cv::line(hist, cv::Point(cross[i - 1].first * (hist_w / cross.size()), hist_h - cross[i - 1].second), cv::Point(cross[i].first * (hist_w / cross.size()), hist_h - cross[i].second), cv::Scalar(0, 0, 255), 1, 1, 0);

        if (cross[max].second < cross[i].second)
        {
            max = i;
        }
    }

    cv::putText(hist, std::to_string(cross[max].first) + ", " + std::to_string(cross[max].second) + "%", cv::Point(cross[max].first * (hist_w / cross.size()), hist_h - cross[max].second - 20), cv::FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255), 1, 1);

    cv::resize(hist, hist, cv::Size(500, 500));

    cv::imshow("hist", hist);
    cv::waitKey(0);
}

void show_data(const Data &data)
{
    int h = 500;
    int w = 500;
    cv::Mat mat(h, w, CV_8UC3, cv::Scalar(0, 0, 0));

    for(size_t i = 0; i < data.size(); ++i)
    {
        cv::Point point((int)((data[i].features()[0] + 1) * 200), (int)((data[i].features()[1] + 1) * 200));

        cv::Scalar color = (data[i].label() == 0) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

        cv::line(mat, point, point, color);
    }

    cv::line(mat, cv::Point(230, 230), cv::Point(230, 230), cv::Scalar(255, 0, 255));

    cv::imshow("data", mat);
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage <file path to data file>" << std::endl;
        return -1;
    }

    srand(time(nullptr));

    Data data;
    data.read(argv[1]);

    show_data(data);

    int max_k = 15;
    int steps = 10;

    std::vector<std::pair<int, int>> hist_acc(max_k, std::make_pair(0, 0));
    std::vector<std::pair<int, int>> hist_cross(max_k, std::make_pair(0, 0));

    for(int step = 0; step < steps; ++step)
    {
        Data test_set, train_set;
        data.split_for_test(0.8, train_set, test_set);

        for(int k = 1; k <= max_k; ++k)
        {
            std::cout << "K:" << k << std::endl;

            double res = cross_validation(train_set, k, 5, 1);
            hist_cross[k - 1].first = k;
            hist_cross[k - 1].second += (int)res;

            Model model;
            model.train(train_set, k);

            std::cout << "TOTAL" << std::endl;

            auto result = model.test(test_set);

            res = print_result(result);
            hist_acc[k - 1].first = k;
            hist_acc[k - 1].second += (int)res;
        }
    }
    std::cout.flush();

    for(auto &p : hist_acc)
    {
        p.second /= steps;
    }
    for(auto &p : hist_cross)
    {
        p.second /= steps;
    }

    show_histogram(hist_acc, hist_cross);

    return 0;
}

