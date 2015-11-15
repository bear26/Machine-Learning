#ifndef TREE_H
#define TREE_H

#include <memory>

#include "data.h"

class Tree
{
public:
    Tree();

    static std::shared_ptr<Tree> train(const Data &data, const std::vector<int> &mask);

    int predict(const Object &obj) const;

private:
    int num_split_features_;
    double seporator_;

    int label_;

    std::shared_ptr<Tree> left_;
    std::shared_ptr<Tree> right_;

    void train_(const Data &data, const std::vector<int> &mask);

    void get_best_split_(const Data &data, const std::vector<int> &mask);

    bool concept_true_(const Object &obj) const;

    void split_(const Data &data, Data &left, Data &right) const;
};

#endif // TREE_H
