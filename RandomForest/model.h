#ifndef MODEL
#define MODEL

#include <memory>

#include "data.h"
#include "tree.h"

class Model
{
public:
    Model();

    void train(const Data &data);

    int predict(const Object &object) const;

    //return vector pair(real label, prediction label)
    std::vector<std::pair<int, int>> test(const Data &data) const;

private:
    std::vector<std::shared_ptr<Tree>> forest_;
};

double print_result(const std::vector<std::pair<int, int>> &result);


#endif // MODEL

