//////////////////////////////////////////////////////////////////////////
// Author		:	Michał Bednarek
// Email		:	michal.gr.bednarek@doctorate.put.poznan.pl
// Organization	:	Poznan University of Technology
// Date			:	2017
//////////////////////////////////////////////////////////////////////////

#ifndef _ALGORITHM_H_
#define _ALGORITHM_H_

#include <iostream>
#include <memory>
#include <vector>
#include <map>

// forward declaration
class Reconstructor;

namespace put
{
class Algorithm
{
public:

    // Optimizer type
    enum Type {
        SA
    };

    // overloaded constructor
    Algorithm(const std::string _name, Type _type) : type(_type), name(_name) {}

    // cannot be const reference -> we need to mutate Reconstructor class internal matrices during optimization
    virtual void Run(Reconstructor& rec) = 0;
    virtual void Init() = 0;

    virtual std::vector<double> GetOptimalParameter() = 0;
    virtual double GetOptimalSolution() = 0;
    virtual void Reset(const double &resetValue) = 0;
    virtual ~Algorithm() {}

protected:
    // Optimizer type
    Type type;

    // Optimizer name
    const std::string name;

};
}

#endif
