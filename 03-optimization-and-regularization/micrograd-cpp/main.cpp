#include <iostream>

#include "engine.h"
#include "nn.h"

int main(){
    //  inputs x1, x2
    Value x1(2.0);
    Value x2(0.0);

    // weights w1,w2
    Value w1(-3.0);
    Value w2(1.0);
    // bias
    Value b(6.8813735870195432);
    Value x1w1 = x1 * w1;
    Value x2w2 = x2 * w2;
    Value x1w1x2w2 = x1w1 + x2w2;
    Value n = x1w1x2w2 + b;
    Value o = n.tanh();

    auto print = [&](){
        std::cout << "x1: " << x1 << std::endl;
        std::cout << "x2: " << x2 << std::endl;
        std::cout << "w1: " << w1 << std::endl;
        std::cout << "w2: " << w2 << std::endl;
        std::cout << "x1w1: " << x1w1 << std::endl;
        std::cout << "x2w2: " << x2w2 << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << "x1w1x2w2: " << x1w1x2w2 << std::endl;
        std::cout << "n: " << n << std::endl;
        std::cout << "o: " << o << std::endl;
        std::cout << "============================" << std::endl;
        std::cout << std::endl;
    };

    print();

    o.backward(); print();

    return 0;
}
