//////////////////////////////////////////////////////////////////////////
// Author		:	Michał Bednarek
// Email		:	michal.gr.bednarek@doctorate.put.poznan.pl
// Organization	:	Poznan University of Technology
// Date			:	2017
//////////////////////////////////////////////////////////////////////////

#ifndef _SIM_ANNEALING_
#define _SIM_ANNEALING_

#include <math.h>
#include <iostream>
#include <cstdlib>
#include <memory>
#include <vector>
#include <map>

#include "Alghoritm.h"

/* function for create object */
Alghoritm* createSA(const int &min, const int &max, const int& innerLoopIterations, const int &numberOfParams);

/* SA class */
class SimulatedAnnealing : public Alghoritm
{
public:
    SimulatedAnnealing(const int &min, const int &max, const int& innerLoopIterations, const int &numberOfParams);
    ~SimulatedAnnealing(){}

    // sets up parameters in simulated annealing alghoritm
    void Init();

    // runs alghoritm with specified parameters
    void Run(Reconstructor *rec);

    // function to be optimized
    double OptimizationFunction();

    /* getters */
    std::vector<double> GetOptimalParameter();
    double GetOptimalSolution();

    double GetRandomNumber(const double& numMin, const double& numMax);

    /* Proper create */
    typedef std::unique_ptr<SimulatedAnnealing> Ptr;

    void Reset(const double& resetValue);


private:
    std::vector<double> params, optimalParameters;
    double optimalSolution;
    double T;
    double decreaseFactor, influenceFactor;
    double tStart, tEnd;
    const double min, max, innerLoopIterator, numberOfParams;
    mutable int counter;

private:
    void UpdateTemperature();
    void UpdateRandomParams(const double &min, const double &max);
    void SetupParams(std::vector<double>& initValues);
    void UpdateOptimalParams();
    double AcceptCondition(const double &solutionEnergy);
};

#endif
