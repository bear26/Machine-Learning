#ifndef MODEL
#define MODEL

#include "data.h"

class Model
{
public:
    Model();

    void train(const Data &data);

    int predict(const Object &object) const;

    //return vector pair(real label, prediction label)
    std::vector<std::pair<int, int>> test(const Data &data) const;

private:
    std::vector<double> w_;
};

double print_result(const std::vector<std::pair<int, int>> &result);
double cross_validation(const Data &data, int folder, int t);


#endif // MODEL

