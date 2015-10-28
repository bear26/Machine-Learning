#include <iostream>
#include <fstream>
#include <algorithm>
#include <fstream>

#include <boost/filesystem.hpp>

#include "data.h"
#include "model.h"

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage <file path to data file>" << std::endl;
        return -1;
    }

    srand(time(nullptr));

    std::fstream stream(argv[1]);
    std::string line;

    double ans = 0;

    while(std::getline(stream, line))
    {
        std::string path = (boost::filesystem::path(argv[1]).parent_path() / boost::filesystem::path(line)).string();

        std::cout << "Testing " << line << std::endl;

         Data data;
         data.read(path);

         Data train_set, test_set;

         data.split_for_test(0.8, train_set, test_set);

         Model model;

         cross_validation(train_set, 5, 1);

         model.train(train_set);

         ans += print_result(model.test(test_set));

         std::cout << std::endl;
    }

    std::cout << "MEAN :" << ans / 10 << std::endl;

    return 0;
}

