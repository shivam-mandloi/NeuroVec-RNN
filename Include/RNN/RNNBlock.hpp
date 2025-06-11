#pragma once

#include "Linear.hpp"
#include "Softmax.hpp"
#include "Relu.hpp"


class RNNBlock
{
public:
    RNNBlock(int hiddenDim) : U(hiddenDim,hiddenDim), W(hiddenDim, hiddenDim){}
    NeuroVec<NeuroVec<double>> Forward(NeuroVec<NeuroVec<double>> &input, NeuroVec<NeuroVec<double>> &hidden)
    {
        NeuroVec<NeuroVec<double>> output = W.Forward(input);
        hidden = U.Forward(hidden);
        hidden = mat2matAdd<double>(output, hidden);
        hidden = rl.Forward(hidden);
        return hidden;
    }
    NeuroVec<NeuroVec<double>> Backward(NeuroVec<NeuroVec<double>> &prevGrad, NeuroVec<NeuroVec<double>> &hiddenGrad)
    {
        // Here we are using linear layer to update and get the derivative of input.
        // we are changing hidden grad.
        hiddenGrad = mat2matAdd<double>(prevGrad, hiddenGrad);
        hiddenGrad = rl.Backward(hiddenGrad);
        NeuroVec<NeuroVec<double>> nextGrad = W.Backward(hiddenGrad);
        hiddenGrad = U.Backward(hiddenGrad);
        return nextGrad;
    }
private:
    Linear U, W;
    Relu rl;
};