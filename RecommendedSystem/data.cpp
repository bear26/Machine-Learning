#include "data.h"

#include <fstream>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

template <typename T>
void from_comma_separated(const std::string &s, std::vector<T> &val)
{
    std::string copy_s = s;
    std::vector<std::string> v;
    boost::trim(copy_s);
    boost::split(v, copy_s, boost::is_any_of(","));

    val.resize(v.size());
    std::transform(v.begin(), v.end(), val.begin(), &boost::lexical_cast<T, std::string>);
}

Data::Data(const std::string &filename)
    :mu_(0)
{
    std::fstream stream(filename);
    std::string line;
    std::getline(stream, line);

    while(std::getline(stream, line))
    {
        std::vector<long long> values;
        from_comma_separated(line, values);

        if (users_.find(values[0]) == users_.end())
        {
            users_[values[0]] = users_.size() - 1;
        }

        if (movies_.find(values[1]) == movies_.end())
        {
            movies_[values[1]] = movies_.size() - 1;
        }

        ratings_[std::make_pair(users_[values[0]], movies_[values[1]])] = values[2];

        mu_ += values[2];
    }

    mu_ /= ratings_.size();

    std::vector<int> count_rating_u(users_.size(), 0);
    std::vector<int> count_rating_m(movies_.size(), 0);

    mean_dev_users_.resize(users_.size(), 0);
    mean_dev_movies_.resize(movies_.size(), 0);

    for(auto it : ratings_)
    {
        count_rating_u[it.first.first]++;
        count_rating_m[it.first.second]++;

        mean_dev_users_[it.first.first] += it.second - mu_;
        mean_dev_movies_[it.first.second] += it.second - mu_;
    }

    for(size_t i = 0; i < mean_dev_users_.size(); ++i)
    {
        mean_dev_users_[i] /= count_rating_u[i];
    }

    for(size_t i = 0; i < mean_dev_movies_.size(); ++i)
    {
        mean_dev_movies_[i] /= count_rating_m[i];
    }
}
