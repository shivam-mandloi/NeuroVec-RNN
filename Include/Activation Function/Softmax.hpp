#pragma once

#include "NeuroVec.hpp"
#include "HelpingFunc.hpp"
#include <cmath>

class Sofmax
{
public:

    NeuroVec<NeuroVec<double>> Forward(NeuroVec<NeuroVec<double>> &input)
    {
        NeuroVec<NeuroVec<double>> copyInput = CopyMatrix<double>(input);
        SoftmaxCalculate(copyInput);
        savedProb = CopyMatrix<double>(copyInput);
        return copyInput;
    }

    NeuroVec<NeuroVec<double>> Backward(NeuroVec<NeuroVec<double>> &prevGrad)
    {
        return SoftmaxDerivative(prevGrad, savedProb);
    }
private:
    NeuroVec<NeuroVec<double>> savedProb;
};