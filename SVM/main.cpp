#include <iostream>
#include <fstream>
#include <algorithm>

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

    Data data;
    data.read(argv[1]);

    int steps = 1;

    for(int step = 0; step < steps; ++step)
    {
        Data test_set, train_set;
        data.split_for_test(0.8, train_set, test_set);

        double res = cross_validation(train_set, 5, 1);

        Model model;
        model.train(train_set, 10);

        std::cout << "TOTAL" << std::endl;

        auto result = model.test(test_set);

        print_result(result);
    }

    std::cout.flush();

    return 0;
}

