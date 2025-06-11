#pragma once

#include "RNNBlock.hpp"
#include <vector>

class RNN
{
public:
    RNN(int inputDim, int outputDim, int hiddenDim): block(hiddenDim), liInput(inputDim, hiddenDim), liOutput(hiddenDim, outputDim)
    {

    }

    void Forward(std::vector<NeuroVec<NeuroVec<double>>> &inputSeq)
    {
        for(int i = 0; i < inputSeq.size(); i++)
        {
            
        }
    }

private:
    RNNBlock block;
    Linear liInput, liOutput;
};