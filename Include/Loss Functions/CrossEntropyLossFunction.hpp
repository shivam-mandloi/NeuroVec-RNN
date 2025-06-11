#pragma once

#include "NeuroVec.hpp"
#include "HelpingFunc.hpp"

class CrossEntropy
{
public:    
    NeuroVec<double> Forward(NeuroVec<NeuroVec<double>> &predicted, NeuroVec<NeuroVec<double>> &groundTruth)
    {
        prevInput = CopyMatrix<double>(predicted);
        prevGroundTruth = CopyMatrix<double>(groundTruth);
        return FindCrossLoss<double>(predicted, groundTruth);
    }

    NeuroVec<NeuroVec<double>> Backward()
    {
        return CrossBackProp<double>(prevInput, prevGroundTruth);
    }
private:
    NeuroVec<NeuroVec<double>> prevInput, prevGroundTruth;
};