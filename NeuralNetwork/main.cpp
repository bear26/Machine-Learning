#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "data.h"
#include "model.h"

void show_error(const std::vector<std::pair<int, int>> &ans, const Data &data)
{
    cv::Mat_<double> cv_mat(400, 1500, 0.0);
    std::vector<int> count(10, 0);

    auto create_mat = [&cv_mat, &count]()
    {
        cv_mat = cv::Mat_<double>::zeros(400, 1500);

        for(size_t i = 0; i < count.size(); ++i)
        {
            cv::putText(cv_mat, std::to_string(i), cv::Point(5, i * 40 + 20), cv::FONT_ITALIC, 0.6, cv::Scalar(255, 255, 255), 1, 1);
        }

        cv::line(cv_mat, cv::Point(25, 0), cv::Point(25, cv_mat.rows), cv::Scalar(255, 255, 255), 1, 1, 0);

        std::fill(count.begin(), count.end(), 0);
    };

    create_mat();

    for(size_t i = 0; i < ans.size(); ++i)
    {
        auto v = ans[i];

        if (v.first != v.second)
        {
            cv::Rect rect(30 + count[v.second] * 30, 40 * v.second, 28, 28);

            cv::Mat_<double>(28, 28, const_cast<double*>(data[i].features().data())).copyTo(cv_mat(rect));

            if (++count[v.second] >= 49)
            {
                cv::imshow("Mistakes", cv_mat);
                cv::waitKey(0);

                create_mat();
            }
        }
    }

    cv::imshow("Mistakes", cv_mat);
    cv::waitKey(0);
}

cv::Mat_<double> mat;
cv::Mat_<double> train_sample;

bool left_mouse_pressed = false;
bool right_mouse_pressed = false;

cv::Point prev_click(-1, -1);
cv::Point prev_click_right(-1, -1);

Model model_predict;

std::chrono::time_point<std::chrono::system_clock> old_time;

void predict()
{
    std::chrono::duration<double> time = std::chrono::system_clock::now() - old_time;

    if (time.count() > 0.33)
    {
        cv::Mat_<double> resize_mat;
        cv::resize(mat, resize_mat, cv::Size(28, 28));

        int result = model_predict.predict(Object(0, std::vector<double>(resize_mat.begin(), resize_mat.end())));

        cv::Mat_<double> digit(100, 100, 0.0);
        cv::putText(digit, std::to_string(result), cv::Point(25, 75), cv::FONT_ITALIC, 2, cv::Scalar(255, 255, 255), 1, 1);

        cv::imshow("Predict digit", digit);

        old_time = std::chrono::system_clock::now();
    }
}

void onMouse(int event, int x, int y, int flags, void *userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        left_mouse_pressed = true;
    }

    if (event == cv::EVENT_LBUTTONUP)
    {
        left_mouse_pressed = false;
        prev_click = cv::Point(-1, -1);
    }

    if (event == cv::EVENT_RBUTTONDOWN)
    {
        right_mouse_pressed = true;
    }

    if (event == cv::EVENT_RBUTTONUP)
    {
        right_mouse_pressed = false;
        prev_click_right = cv::Point(-1, -1);
        prev_click = cv::Point(-1, -1);
    }

    if (left_mouse_pressed || right_mouse_pressed)
    {
        cv::Point click_point(x, y);

        if (prev_click.x != -1)
        {

            cv::line(mat, prev_click, click_point, (left_mouse_pressed) ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 0), (left_mouse_pressed) ? 50 : 100, 1, 0);
        }

        prev_click = click_point;

        cv::imshow("Predict draw", mat);

        predict();
    }
}

void onMouseTrain(int event, int x, int y, int flags, void *userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        cv::Mat_<double> resize_mat;
        cv::resize(mat, resize_mat, cv::Size(28, 28));

        std::cout << "trained as " << y / 70 << std::endl;

        model_predict.learn_on_object(Object(y / 70, std::vector<double>(resize_mat.begin(), resize_mat.end())));

        predict();
    }
}

void predict_draw(const Model model)
{
    old_time = std::chrono::system_clock::now();

    model_predict = model;
    mat = cv::Mat_<double>(500, 500, 0.0);

    cv::namedWindow("Predict draw");
    cv::namedWindow("Train");

    cv::setMouseCallback("Predict draw", onMouse);
    cv::setMouseCallback("Train", onMouseTrain);

    train_sample = cv::Mat_<double>(700, 80);
    for(int i = 0; i < 10; ++i)
    {
        cv::putText(train_sample, std::to_string(i), cv::Point(20, i * 70 + 60), cv::FONT_ITALIC, 2, cv::Scalar(255, 255, 255), 1, 1);
    }

    cv::imshow("Predict draw", mat);
    cv::imshow("Train", train_sample);

    cv::waitKey(0);
}

int main(int argc, char* argv[])
{
    if (argc != 5)
    {
        std::cerr << "Usage <train|test|draw> <dataset_im> <dataset_labels> <weight for test>" << std::endl;
        return -1;
    }

    srand(time(nullptr));

    if (std::string(argv[1]) == "train")
    {
        Data data;
        data.read(argv[2], argv[3]);

        Model model;

        model.train(data);

        model.save(argv[4]);
    }
    else if (std::string(argv[1]) == "test")
    {
        Data data;
        data.read(argv[2], argv[3]);

        Model model;

        model.load(argv[4]);

        auto result = model.test(data);

        print_result(result);

        show_error(result, data);
    }
    else
    {
        Data data;
        data.read(argv[2], argv[3]);

        Model model;

        model.load(argv[4]);

        predict_draw(model);
    }

    return 0;
}

