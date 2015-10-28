#include "data.h"

#include <fstream>
#include <algorithm>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

Data::Data()
{

}

Data::Data(const std::vector<Object> &objects)
    :objects_(objects)
{

}

void Data::add(const Data &data)
{
    objects_.insert(objects_.end(), data.objects_.begin(), data.objects_.end());
}

void Data::add(const Object &object)
{
    objects_.push_back(object);
}

void Data::read(const std::string &filepath)
{
    std::fstream stream(filepath);
    std::string line;

    //skip first line
    getline(stream, line);

    while(getline(stream, line))
    {
        std::vector<std::string> split_vec;

        boost::trim(line);
        boost::split(split_vec, line, boost::is_any_of(" \t"), boost::token_compress_on);

        std::vector<double> features;

        for(auto &s : split_vec)
        {
            boost::replace_all(s, ",", ".");
            features.push_back(boost::lexical_cast<double>(s));
        }

        int label = features.back();

        if (label == 0)
        {
            label = -1;
        }

        features.pop_back();

        objects_.push_back(Object(label, features));
    }

    std::random_shuffle(objects_.begin(), objects_.end());
}

void Data::split_for_test(double part_for_train, Data &train_set, Data &test_set) const
{
    std::vector<Object> objects = objects_;

    std::sort(objects.begin(), objects.end());

    std::vector<Object> train_s;
    std::vector<Object> test_s;

    for(size_t i = 0; i < objects.size();)
    {
        size_t from = i;

        while(i < objects.size() && objects[i].label() == objects[from].label())
        {
            ++i;
        }

        std::random_shuffle(objects.begin() + from, objects.begin() + i);

        for(size_t j = from; j < i; ++j)
        {
            if (j < from + (i - from) * part_for_train)
            {
                train_s.push_back(objects[j]);
            }
            else
            {
                test_s.push_back(objects[j]);
            }
        }
    }

    std::random_shuffle(train_s.begin(), train_s.end());
    std::random_shuffle(test_s.begin(), test_s.end());

    train_set = Data(train_s);
    test_set = Data(test_s);
}

void Data::split(int folder, std::vector<Data> &data) const
{
    std::vector<Object> objects = objects_;

    std::sort(objects.begin(), objects.end());

    std::vector<std::vector<Object>> sets(folder);
    for(size_t i = 0; i < objects.size();)
    {
        size_t from = i;

        while(i < objects.size() && objects[i].label() == objects[from].label())
        {
            ++i;
        }

        std::random_shuffle(objects.begin() + from, objects.begin() + i);

        for(int k = 0; k < folder; ++k)
        {
            for(size_t j = from + (i - from) * k / (folder); j < from + (i - from) * (k + 1) / (folder); ++j)
            {
                sets[k].push_back(objects[j]);
            }
        }
    }

    data.clear();
    for(auto &vec : sets)
    {
        std::random_shuffle(vec.begin(), vec.end());
        data.push_back(Data(vec));
    }
}
