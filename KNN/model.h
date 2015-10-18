#ifndef MODEL
#define MODEL

#include "data.h"

class Model
{
public:
    Model();

    int predict(const Object &object) const;

    //return vector pair(real label, prediction label)
    std::vector<std::pair<int, int>> test(const Data &data) const;

    //train KNN classificator
    void train(const Data &data, int k);

private:
    Data _data;
    int _k;
};

double print_result(const std::vector<std::pair<int, int>> &result);
double cross_validation(const Data &data, int k, int folder, int t);


#endif // MODEL

