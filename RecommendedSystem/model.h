#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <functional>

#include "data.h"

class Model
{
public:
    Model(int count_feaatures);

    void solve(const Data &data);

    double predict(long long user_hash, long long movid_hash) const;

private:
    int count_features_;

    std::vector<double> bu_;
    std::vector<double> bi_;

    std::vector<std::vector<double>> p_;
    std::vector<std::vector<double>> q_;

    // map from user hash to user_id
    std::map<long long, int> users_;
    // map from movie hash to movie_id
    std::map<long long, int> movies_;

    double predict_(int user_id, int movie_id) const;

    double mu_;

    const double step1 = 0.005;
    const double step2 = 0.005;

    const double lamda1 = 0.02;
    const double lamda2 = 0.02;
};

#endif // MODEL_H
