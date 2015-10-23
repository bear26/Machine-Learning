#ifndef OBJECT
#define OBJECT

#include <vector>
#include <string>

class Object
{
public:
    Object();
    Object(const std::string &filename);

    inline int label() const { return label_; }
    inline const std::vector<int> &subject() const { return subject_; }
    inline const std::vector<int> &body() const { return body_;}

    bool operator <(const Object &object) const;

private:
    int label_;
    std::vector<int> subject_;
    std::vector<int> body_;
};

#endif // OBJECT

