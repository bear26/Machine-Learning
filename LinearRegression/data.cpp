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

void Data::read(const std::string &filepath)
{
    std::fstream stream(filepath);
    std::string line;

    while(std::getline(stream, line))
    {
        std::vector<std::string> split_vec;

        boost::trim(line);
        boost::split(split_vec, line, boost::is_any_of(","), boost::token_compress_on);

        std::vector<double> features;

        for(auto &s : split_vec)
        {
            features.push_back(boost::lexical_cast<double>(s));
        }

        int label = features.back();
        features.pop_back();

        objects_.push_back(Object(label, features));
    }

    std::random_shuffle(objects_.begin(), objects_.end());
}

void Data::split_for_test(double part_for_train, Data &train_set, Data &test_set) const
{
    std::vector<Object> train_s(objects_.begin(), objects_.begin() + objects_.size() * part_for_train);
    std::vector<Object> test_s(objects_.begin() + objects_.size() * part_for_train, objects_.end());

    std::random_shuffle(train_s.begin(), train_s.end());
    std::random_shuffle(test_s.begin(), test_s.end());

    train_set = Data(train_s);
    test_set = Data(test_s);
}
