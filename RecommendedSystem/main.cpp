#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <functional>

#include "model.h"

template <typename T>
void from_comma_separated(const std::string &s, std::vector<T> &val);

void test(const Model &model, const std::string &filename)
{
    std::fstream stream(filename);
    std::string line;
    std::getline(stream, line);

    double ans = 0;
    int count = 0;

    while(std::getline(stream, line))
    {
        std::vector<long long> values;
        from_comma_separated(line, values);

        ans += std::pow(model.predict(values[0], values[1]) - values[2], 2);
        ++count;
    }

    std::cout << "RMSE: " << std::sqrt(ans / count) << std::endl;
}

void predict(const Model &model, const std::string &filename, const std::string &dst)
{
    std::fstream stream(filename);
    std::fstream stream_out(dst, std::ios_base::out);
    std::string line;
    std::getline(stream, line);

    stream_out << "id,rating" << std::endl;

    while(std::getline(stream, line))
    {
        std::vector<long long> values;
        from_comma_separated(line, values);

        stream_out << values[0] << "," << model.predict(values[1], values[2]) << std::endl;
    }
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cerr << "Usage <train filename> <validation filename> <test filename> <result filename>" << std::endl;
        return -1;
    }
    Data data(argv[1]);

    int count_features = 6;

    Model model(count_features);
    model.solve(data);

    test(model, argv[2]);

    predict(model, argv[3], argv[4]);

    return 0;
}

