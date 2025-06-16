#pragma once

#include "NeuroVec.hpp"
#include "HelpingFunc.hpp"
#include <vector>
#include <cmath>


class Sofmax
{
public:
    std::vector<NeuroVec<NeuroVec<double>>> Forward(std::vector<NeuroVec<NeuroVec<double>>> &input)
    {
        std::vector<NeuroVec<NeuroVec<double>>> res;
        for(int i = 0; i < input.size(); i++)
        {
            NeuroVec<NeuroVec<double>> copyInput = CopyMatrix<double>(input[i]);
            SoftmaxCalculate(copyInput);
            res.push_back(copyInput);
            savedProb.push_back(CopyMatrix<double>(copyInput));
        }
        return res;
    }

    std::vector<NeuroVec<NeuroVec<double>>> Backward(std::vector<NeuroVec<NeuroVec<double>>> &prevGrad)
    {
        std::vector<NeuroVec<NeuroVec<double>>> grad;
        for(int i = 0; i < prevGrad.size(); i++)
        {
            grad.push_back(SoftmaxDerivative(prevGrad[i], savedProb[i]));
        }
        return grad;
    }

private:
    std::vector<NeuroVec<NeuroVec<double>>> savedProb;
};