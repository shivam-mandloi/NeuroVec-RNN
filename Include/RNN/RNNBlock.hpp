#pragma once

#include <vector>
#include "Linear.hpp"
#include "Softmax.hpp"
#include "Relu.hpp"


class RNNBlock
{
public:
    RNNBlock(int hiddenDim) : U(hiddenDim,hiddenDim), W(hiddenDim, hiddenDim){}
    NeuroVec<NeuroVec<double>> Forward(NeuroVec<NeuroVec<double>> &input, NeuroVec<NeuroVec<double>> &hidden)
    {        
        wInput.push_back(input);
        NeuroVec<NeuroVec<double>> output = W.Forward(input);
        
        uInput.push_back(hidden);
        hidden = U.Forward(hidden);

        hidden = mat2matAdd<double>(output, hidden);
        
        rlInput.push_back(hidden);
        hidden = rl.Forward(hidden);

        return hidden;
    }
    
    NeuroVec<NeuroVec<double>> Backward(NeuroVec<NeuroVec<double>> &prevGrad, NeuroVec<NeuroVec<double>> &hiddenGrad, int index)
    {
        // Here we are using linear layer to update and get the derivative of input.
        // we are changing hidden grad.

        hiddenGrad = mat2matAdd<double>(prevGrad, hiddenGrad);
        hiddenGrad = rl.Backward(hiddenGrad, rlInput[index]);
        NeuroVec<NeuroVec<double>> nextGrad = W.Backward(hiddenGrad, wInput[index]);
        hiddenGrad = U.Backward(hiddenGrad, uInput[index]);
        return nextGrad;
    }
private:
    Linear U, W;
    Relu rl;
    std::vector<NeuroVec<NeuroVec<double>>> rlInput, wInput, uInput;
};