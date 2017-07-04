#include "../include/SimulatedAnnealing.h"
//////////////////////////////////////////////////////////////////////////
// Author		:	Michał Bednarek
// Email		:	michal.gr.bednarek@doctorate.put.poznan.pl
// Organization	:	Poznan University of Technology
// Date			:	2017
//////////////////////////////////////////////////////////////////////////


/* Properly create object */
SimulatedAnnealing::Ptr SA;

Alghoritm* createSA(const int &min, const int &max, const int& innerLoopIterations, const int &numberOfParams)
{
    SA.reset(new SimulatedAnnealing(min, max, innerLoopIterations, numberOfParams));
    return SA.get();
}

/* SA class: */
SimulatedAnnealing::SimulatedAnnealing(const int &min, const int &max, const int& innerLoopIterations, const int& numberOfParams)
    : Alghoritm("SA", SA), min(min), max(max), innerLoopIterator(innerLoopIterations), numberOfParams(numberOfParams)
{
    srand (time(NULL));
}

void SimulatedAnnealing::Init()
{
    //initialize internal param vectors
    this->params            = std::vector<double>(this->numberOfParams);
    this->optimalParameters = std::vector<double>(this->numberOfParams);

    // specify init parameters
    this->UpdateRandomParams(this->min, this->max);
    this->UpdateOptimalParams();

    // setup alghoritm variables
    this->tStart           = 100;
    this->tEnd             = 10;
    this->T                = this->tStart;
    this->decreaseFactor   = 0.95;
    this->influenceFactor  = 1e-5;
    this->counter          = 0;
}

void SimulatedAnnealing::Run(Reconstructor* rec)
{
    this->optimalSolution = arma::mean(rec->reprojErrors);

    // start alghoritm
    while(this->T > this->tEnd)
    {
        int k = 1;
        while(k < this->innerLoopIterator)
        {
            // randomly update parameters with random number in specified range
            this->UpdateRandomParams(-0.5, 0.5);

            // check outputr
            double tempOutput = rec->adjustFocal(this->params);

            // optimize solution if output is accepted
            if((tempOutput < this->optimalSolution) || this->AcceptCondition(tempOutput) > this->GetRandomNumber(0, 1) )
            {
                this->optimalSolution  = tempOutput;
                this->UpdateOptimalParams();
                if(this->GetOptimalParameter().size() == 1) // we have only one parameter to be optimized
                {
                    rec->updateInternalMatrices(this->GetOptimalParameter()[0]);
                }
            }
            k++;
        }
        this->counter++;
        //std::cout << counter << ": " << T << " " << tEnd << " " << rec->modelCamCamera.getFocal() << std::endl;
        this->UpdateTemperature();
    }
}

std::vector<double> SimulatedAnnealing::GetOptimalParameter()
{
    return this->optimalParameters;
}

double SimulatedAnnealing::GetOptimalSolution()
{
    return this->optimalSolution;
}

double SimulatedAnnealing::GetRandomNumber(const double& numMin, const double& numMax)
{
    double f = static_cast<double>(rand()) / RAND_MAX;
    return numMin + f * (numMax - numMin);
}

double SimulatedAnnealing::AcceptCondition(const double& solutionEnergy)
{
    double p = -(solutionEnergy - this->optimalSolution) / (this->influenceFactor * this->T);
    return pow(exp(1), p);
}

void SimulatedAnnealing::Reset(const double &resetValue)
{
    for(size_t i = 0; i < this->numberOfParams; ++i)
    {
        this->params[i] = resetValue;
        this->optimalParameters[i] = resetValue;
    }
    this->counter = 0;
    this->T = this->tStart;
}

void SimulatedAnnealing::UpdateTemperature()
{
    this->T = this->T * this->decreaseFactor;
}

void SimulatedAnnealing::UpdateRandomParams(const double& min, const double& max)
{
    for(size_t i = 0; i < numberOfParams; ++i)
    {
        this->params[i] += this->GetRandomNumber(min, max);;
    }
}

void SimulatedAnnealing::SetupParams(std::vector<double>& initValues)
{
    if(initValues.size() != this->numberOfParams)
    {
        std::cerr << "Wrong number of init parameters!" << std:: endl;
        abort();
    }
    else
    {
        for(size_t i = 0; i < this->numberOfParams; ++i)
        {
            this->params[i] = initValues[i];
        }
    }
}

void SimulatedAnnealing::UpdateOptimalParams()
{
    for(size_t i = 0; i < this->params.size(); ++i)
    {
        this->optimalParameters[i] = this->params[i];
    }
}
