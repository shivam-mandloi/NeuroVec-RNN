#pragma once

#include "NeuroVec.hpp"
#include "HelpingFunc.hpp"

class CrossEntropy
{
public:    
    std::vector<NeuroVec<double>> Forward(std::vector<NeuroVec<NeuroVec<double>>> &predicted, std::vector<NeuroVec<NeuroVec<double>>> &groundTruth)
    {
        std::vector<NeuroVec<double>> res;
        for(int i = 0; i < predicted.size(); i++)
        {
            res.push_back(FindCrossLoss<double>(predicted[i], groundTruth[i]));
        }
        return res;
    }

    std::vector<NeuroVec<NeuroVec<double>>> Backward(std::vector<NeuroVec<NeuroVec<double>>> &prevInput, std::vector<NeuroVec<NeuroVec<double>>> &prevGroundTruth)
    {
        std::vector<NeuroVec<NeuroVec<double>>> grad;
        for(int i = 0; i < prevInput.size(); i++)
        {
            grad.push_back(CrossBackProp<double>(prevInput[i], prevGroundTruth[i]));
        }
        return grad;
    }

};