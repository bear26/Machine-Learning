#ifndef OBJECT
#define OBJECT

#include <vector>

class Object
{
public:
    Object();
    Object(int label, const std::vector<double> features);

    inline int label() const { return _label; }
    inline const std::vector<double> &features() const { return _features; }

    double distance(const Object &obj) const;
    double metric(const Object &obj, double max_distance) const;

    bool operator <(const Object &object) const;

private:
    int _label;
    std::vector<double> _features;
};

#endif // OBJECT

