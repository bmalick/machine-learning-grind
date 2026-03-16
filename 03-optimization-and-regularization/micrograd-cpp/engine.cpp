#include <iomanip>
#include <iostream>
#include <functional>
#include <cmath>

#include "engine.h"

Value::Value(){}

Value::~Value(){}

Value::Value(double data_param, std::vector<Value*> children, std::string op_param): data(data_param), _op(op_param), _prev(children.begin(), children.end()) {}

std::ostream& operator<<(std::ostream& out, const Value& v){
    out << "Value(data=" << std::setprecision(5) << v.data << ", grad=" << std::setprecision(5) << v.grad << ")";
    return out;
}


Value Value::operator+(Value& v) {
    Value out(data+v.data, {this, &v}, "+");

    out._backward = [this, &v, &out](){
        grad += 1.0 * out.grad;
        v.grad += 1.0 * out.grad;
        std::cout << "+" << std::endl;
    };
    return out;
}


Value Value::operator*(Value& v) {
    Value out(data*v.data, {this, &v}, "*");

    out._backward = [this, &v, &out](){
        grad += v.data * out.grad;
        v.grad += data * out.grad;
        std::cout << "*" << std::endl;
    };
    return out;
}


Value Value::tanh(){
    double t = (std::exp(2*data) - 1) / (std::exp(2*data) + 1);
    Value out(t, {this}, "tanh");
    out._backward = [this, &out](){
        grad += (1-std::pow(out.data,2)) * out.grad;
        std::cout << "tanh" << std::endl;
    };
    return out;
}

void Value::backward(){
    std::vector<Value*> topo {};
    std::set<Value*> visited = {};
    std::function<void(Value*)> build_topo;
    build_topo = [&](Value* v){
        if (visited.find(v)==visited.end()){
            visited.insert(v);
            for (auto child : v->_prev){
                build_topo(child);
            }
            topo.push_back(v);
        }
    };

    build_topo(this);
    grad = 1.0;
    for (int i=topo.size()-1;i>=0; i--){
        topo[i]->_backward();
    }
}
