#pragma once

#include "NeuroVec.hpp"
#include "HelpingFunc.hpp"

class MSE
{
public:
    NeuroVec<double> Forward(NeuroVec<NeuroVec<double>> &predicted, NeuroVec<NeuroVec<double>> &groundTruth)
    {
        prevInput = CopyMatrix<double>(predicted);
        prevGroundTruth = CopyMatrix<double>(groundTruth);
        return MseForward(predicted, groundTruth);
    }

    NeuroVec<NeuroVec<double>> Backward()
    {
        return MseBackProp(prevInput, prevGroundTruth);
    }
private:
    NeuroVec<NeuroVec<double>> prevInput, prevGroundTruth;
};