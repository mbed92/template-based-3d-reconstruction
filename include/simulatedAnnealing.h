#ifndef _SIM_ANNEALING_
#define _SIM_ANNEALING_

#include <math.h>
#include <iostream>
#include <cstdlib>

class SimulatedAnnealing
{
public:
    SimulatedAnnealing()
    {
        srand (time(NULL));
    }
    ~SimulatedAnnealing(){}

    void init(const int &min, const int &max);
    void reset(double& focal);
    double accept(double &solutionEnergy, double &neighbourEnergy, double &temperature);
    double getRandom(double numMin, double numMax);
    void updateTemperature();
    double updateParam(const double &min, const double &max);

    double param;
    double optimalSolution;
    double a, d;
    double T, tI, tF;
    double minim;
    double currentOutput;
    double counter;
};

#endif
