#include "../include/simulatedAnnealing.h"

void SimulatedAnnealing::init(const int& min, const int& max)
{
    this->param = getRandom(min, max);
    this->optimalSolution = this->param;
    this->tI = 100;
    this->tF = 10;
    this->a = 0.95;
    this->d= 1e-5;
    this->T = tI;
    this->counter = 0;
}

void SimulatedAnnealing::reset(double& focal)
{
    this->param = focal;
    this->optimalSolution = focal;
    this->counter = 0;
    this->T = tI;
}

double SimulatedAnnealing::accept(double& solutionEnergy, double& neighbourEnergy, double& temperature)
{
    double p = -(solutionEnergy - neighbourEnergy) / (this->d * temperature);
    return pow(exp(1), p);
}

double SimulatedAnnealing::getRandom(double numMin, double numMax)
{
    double f = static_cast<double>(rand()) / RAND_MAX;
    return numMin + f * (numMax - numMin);
}

void SimulatedAnnealing::updateTemperature()
{
    this->T = this->T * this->a;
}

double SimulatedAnnealing::updateParam(const double& min, const double& max)
{
    this->param = this->param + this->getRandom(min, max);
    return this->param;
}
