#ifndef DATA_H
#define DATA_H

#include <map>
#include <string>
#include <vector>

class Data
{
public:
    Data(const std::string &filename);

    double get_mean_dev_for_user(int user_id) const { return mean_dev_users_[user_id]; }
    double get_mean_dev_for_movie(int movie_id) const{ return mean_dev_movies_[movie_id]; }

    double get_mean_all_movies() const { return mu_; }

    size_t get_count_users() const { return users_.size(); }
    size_t get_count_movies() const { return movies_.size(); }

    std::map<long long, int> get_users_map() const { return users_; }
    std::map<long long, int> get_movies_map() const { return movies_; }

    std::map<std::pair<int, int>, int>::iterator begin() { return ratings_.begin(); }
    std::map<std::pair<int, int>, int>::iterator end() { return ratings_.end(); }

    std::map<std::pair<int, int>, int>::const_iterator begin() const { return ratings_.begin(); }
    std::map<std::pair<int, int>, int>::const_iterator end() const { return ratings_.end(); }

private:
    // map from user hash to user_id
    std::map<long long, int> users_;
    // map from movie hash to movie_id
    std::map<long long, int> movies_;

    std::map<std::pair<int, int>, int> ratings_;

    // stat
    double mu_;
    std::vector<double> mean_dev_users_;
    std::vector<double> mean_dev_movies_;
};

#endif // DATA_H
