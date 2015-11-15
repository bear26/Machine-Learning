#include <iostream>
#include <fstream>
#include <algorithm>

#include "data.h"
#include "model.h"

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage <filepath to data_train without extension> <filepath to data_test without extension> " << std::endl;
        return -1;
    }

    srand(time(nullptr));

    Data train_set, test_set;
    train_set.read(argv[1]);
    test_set.read(argv[2]);

    Model model;
    model.train(train_set);

    auto result = model.test(test_set);

    print_result(result);

    return 0;
}

