#ifndef ENGINE
#define ENGINE

#include <iostream>
#include <set>
#include <vector>
#include <functional>

class Value{
    friend std::ostream& operator<<(std::ostream& out, const Value& v);

    public:
        Value();
        ~Value();
        Value(double data_param, std::vector<Value*> children = {}, std::string _op = "");
        // Value(double data_param, std::set<Value*> _children);

        Value operator+(Value& v);
        Value operator*(Value& v);
        Value tanh();
        void backward();





    // private:
    public:
        double data;
        double grad = 0.0;
        std::set<Value*> _prev;
        std::string _op;
        std::function<void()> _backward = [](){};

};

#endif
