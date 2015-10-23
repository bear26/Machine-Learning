#include "object.h"

#include <algorithm>
#include <cmath>
#include <fstream>

#include <boost/algorithm/string.hpp>

Object::Object()
{

}

Object::Object(const std::string &filename)
{
    // 0 - spam; 1 - legit
    label_ = (filename.find("legit") != std::string::npos);

     std::fstream stream(filename);

    std::string line;

    //read subject
    std::getline(stream, line);
    std::vector<std::string> vec_str;

    boost::split(vec_str, line, boost::is_any_of(" "), boost::token_compress_on);

    for(size_t i = 1; i < vec_str.size(); ++i)
    {
        if (!vec_str[i].empty())
        {
            subject_.push_back(std::stoi(vec_str[i]));
        }
    }

    //read body
    while(std::getline(stream, line))
    {
        boost::split(vec_str, line, boost::is_any_of(" "), boost::token_compress_on);

        for(auto s : vec_str)
        {
            if (!s.empty())
            {
                body_.push_back(std::stoi(s));
            }
        }
    }
}

bool Object::operator <(const Object &object) const
{
    return label_ < object.label_;
}


