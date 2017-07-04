//////////////////////////////////////////////////////////////////////////
// Author		:	Michał Bednarek
// Email		:	michal.gr.bednarek@doctorate.put.poznan.pl
// Organization	:	Poznan University of Technology
// Date			:	2017
//////////////////////////////////////////////////////////////////////////

#ifndef _ALGHORITM_H_
#define _ALGHORITM_H_

#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include "Reconstructor.h"

class Reconstructor;

class Alghoritm
{
public:

    // Optimizer type
    enum Type {
        SA
    };

    // overloaded constructor
    Alghoritm(const std::string _name, Type _type) : type(_type), name(_name) {}

    virtual void Run(Reconstructor* rec) = 0;
    virtual void Init() = 0;

    virtual std::vector<double> GetOptimalParameter() = 0;
    virtual double GetOptimalSolution() = 0;
    virtual void Reset(const double &resetValue) = 0;
    virtual ~Alghoritm() {}
protected:
    // Optimizer type
    Type type;

    // Optimizer name
    const std::string name;

};

#endif
