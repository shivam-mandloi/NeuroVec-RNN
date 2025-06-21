#pragma once

#include "RNNBlock.hpp"
#include <vector>

class RNN
{
public:
    RNN(int _hiddenDim): block(_hiddenDim)
    {
        hiddenDim = _hiddenDim;
    }

    std::vector<NeuroVec<NeuroVec<double>>> Forward(std::vector<NeuroVec<NeuroVec<double>>> &inputSeq)
    {
        NeuroVec<NeuroVec<double>> hidden = CreateMatrix<double>(inputSeq[0].len, hiddenDim, 0);
        std::vector<NeuroVec<NeuroVec<double>>> res;
        NeuroVec<NeuroVec<double>> output;

        for(int i = 0; i < inputSeq.size() - 1; i++)
        {
            output = block.Forward(inputSeq[i], hidden);
            res.push_back(output);
        }
        return res;
    }

    std::vector<NeuroVec<NeuroVec<double>>> Backward(std::vector<NeuroVec<NeuroVec<double>>> &lossGrad)
    {
        NeuroVec<NeuroVec<double>> hiddenGrad = CreateMatrix<double>(lossGrad[0].len, hiddenDim, 0);
        std::vector<NeuroVec<NeuroVec<double>>> res;
        for(int i = lossGrad.size() - 1; i > -1; i--)
        {
            NeuroVec<NeuroVec<double>> resGrad = block.Backward(lossGrad[i], hiddenGrad, i);
            res.push_back(resGrad);
        }
        return res;
    }

private:
    RNNBlock block;
    int hiddenDim;
};