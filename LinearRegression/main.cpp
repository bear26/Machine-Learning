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

    Data train_data, test_data;
    data.split_for_test(0.8, train_data, test_data);

    Model model;
    model.train(train_data);

    double ans = model.predict(test_data);

    std::cout << "Mean Error: " << ans << std::endl;

    while(true)
    {
        std::vector<double> features(2);

        std::cin >> features[0] >> features[1];

        std::cout << model.predict(features) << std::endl;
    }


    return 0;
}

