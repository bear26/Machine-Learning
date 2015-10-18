#ifndef DATA
#define DATA

#include "object.h"

#include <vector>
#include <string>

class Data
{
public:
    Data();
    Data(const std::vector<Object> &objects);

    void add(const Data &data);

    void read(const std::string &filepath);

    void split_for_test(double part_for_train, Data &train_set, Data &test_set) const;
    void split(int folder, std::vector<Data> &data) const;

    inline size_t size() const { return _objects.size(); }
    inline Object& operator [](int i) { return _objects[i]; }
    inline const Object& operator [](int i) const { return _objects[i]; }

private:
    std::vector<Object> _objects;
};


#endif // DATA

