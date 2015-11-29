#include "model.h"

#include <numeric>
#include <algorithm>

template <typename T>
std::vector<T> operator *(T val, const std::vector<T> &v)
{
    std::vector<T> ans(v.size());
    std::transform(v.begin(), v.end(), ans.begin(), [&val](T v){ return val * v; });

    return ans;
}

template <typename T>
std::vector<T> operator -(const std::vector<T> &v1, const std::vector<T> &v2)
{
    std::vector<T> ans(v1.size());
    std::transform(v1.begin(), v1.end(), v2.begin(), ans.begin(), [](T val1, T val2){ return val1 - val2; });

    return ans;
}

template <typename T>
std::vector<T> operator +(const std::vector<T> &v1, const std::vector<T> &v2)
{
    std::vector<T> ans(v1.size());
    std::transform(v1.begin(), v1.end(), v2.begin(), ans.begin(), [](T val1, T val2){ return val1 + val2; });

    return ans;
}


Model::Model(int count_feaatures)
    :count_features_(count_feaatures)
{

}

void Model::solve(const Data &data)
{
    users_ = data.get_users_map();
    movies_ = data.get_movies_map();

    bu_.resize(data.get_count_users());
    bi_.resize(data.get_count_movies());

    for(size_t i = 0; i < bu_.size(); ++i)
    {
        bu_[i] = data.get_mean_dev_for_user(i);
    }

    for(size_t i = 0; i < bi_.size(); ++i)
    {
        bi_[i] = data.get_mean_dev_for_movie(i);
    }

    p_.resize(data.get_count_users(), std::vector<double>(count_features_, 0.2));
    q_.resize(data.get_count_movies(), std::vector<double>(count_features_, 0.2));

    mu_ = data.get_mean_all_movies();

    std::vector<std::pair<std::pair<int, int>, int>> ratings(data.begin(), data.end());
    std::random_shuffle(ratings.begin(), ratings.end());

    //maybe not once
    for(auto it : ratings)
    {
        int user_id = it.first.first;
        int movie_id = it.first.second;

        double e = it.second - predict_(user_id, movie_id);

        bu_[user_id] += step1 * (e - lamda1 * bu_[user_id]);
        bi_[movie_id] += step1 * (e - lamda1 * bi_[movie_id]);

        q_[movie_id] = q_[movie_id] + step2 * (e * p_[user_id] - lamda2 * q_[movie_id]);
        p_[user_id] = p_[user_id] + step2 * (e * q_[movie_id] - lamda2 * p_[user_id]);
    }
}

double Model::predict(long long user_hash, long long movie_hash) const
{
    if (users_.find(user_hash) == users_.end() || movies_.find(movie_hash) == movies_.end())
    {
        return 3;
    }

    return predict_(users_.at(user_hash), movies_.at(movie_hash));
}

double Model::predict_(int user_id, int movie_id) const
{
    return mu_ + bu_[user_id] + bi_[movie_id] + std::inner_product(p_[user_id].begin(), p_[user_id].end(), q_[movie_id].begin(), 0.0);
}
