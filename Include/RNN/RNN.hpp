#pragma once

#include "RNNBlock.hpp"
#include <vector>

class RNN
{
public:
    RNN(int _inputDim, int _outputDim, int _hiddenDim): block(_hiddenDim), liInput(_inputDim, _hiddenDim), liOutput(_hiddenDim, _outputDim)
    {
        hiddenDim = hiddenDim;
    }

    std::vector<NeuroVec<NeuroVec<double>>> Forward(std::vector<NeuroVec<NeuroVec<double>>> &inputSeq)
    {
        NeuroVec<NeuroVec<double>> hidden = CreateMatrix<double>(inputSeq[0].len, hiddenDim, 0);
        std::vector<NeuroVec<NeuroVec<double>>> res;
        NeuroVec<NeuroVec<double>> output;
        for(int i = 0; i < inputSeq.size(); i++)
        {
            output = liInput.Forward(inputSeq[i]);
            output = block.Forward(output, hidden);
            output = liOutput.Forward(output);
            res.push_back(output);
        }
        return res;
    }

    NeuroVec<NeuroVec<double>> Backward(std::vector<NeuroVec<NeuroVec<double>>> lossGrad)
    {
        NeuroVec<NeuroVec<double>> hiddenGrad = CreateMatrix<double>(lossGrad[0].len, hiddenDim, 0);
        for(int i = lossGrad.size() - 1; i > -1; i++)
        {
            NeuroVec<NeuroVec<double>> dldht = liInput.Backward(lossGrad[i]);
            dldht = block.Backward(dldht, hiddenGrad);
            liOutput.Backward(dldht);
        }
    }

private:
    RNNBlock block;
    Linear liInput, liOutput;
    int hiddenDim;
};