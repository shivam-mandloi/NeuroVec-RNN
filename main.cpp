#include <vector>

#include "CrossEntropyLossFunction.hpp"
#include "MSE.hpp"
#include "Relu.hpp"
#include "Softmax.hpp"
#include "SGD.hpp"
#include "Linear.hpp"

using namespace std;

struct nn
{
    Linear li1, li2, li3;
    Relu rl1, rl2, rl3;
    Sofmax sf;
    MSE mseLoss;
    nn():li1(9, 64), li2(64, 32), li3(32, 1){}
    NeuroVec<NeuroVec<double>> Forward(NeuroVec<NeuroVec<double>> input)
    {
        input = li1.Forward(input);
        input = rl1.Forward(input);
        input = li2.Forward(input);
        input = rl2.Forward(input);
        input = li3.Forward(input);
        return input;
    }

    void BackPropagate()
    {
        NeuroVec<NeuroVec<double>> grad = mseLoss.Backward();
        // grad = rl3.Backward(grad);
        grad = li3.Backward(grad);
        grad = rl2.Backward(grad);
        grad = li2.Backward(grad);
        grad = rl1.Backward(grad);
        li1.Backward(grad);
    }

    double loss(NeuroVec<NeuroVec<double>> input, NeuroVec<double> actual, bool check = 1)
    {
        NeuroVec<NeuroVec<double>> groundTruth = CreateMatrix<double>(input.len, input[0].len, 0);
        for(int i = 0; i < actual.len; i++)
        {
            groundTruth[i][0] = actual[i];
        }
        NeuroVec<double> loss = mseLoss.Forward(input, groundTruth);
        double totalLoss = 0;
        for(int i = 0; i < loss.len; i++)
        {
            totalLoss += loss[i];
        }       
        if(check)
            BackPropagate();
        return totalLoss/loss.len;
    }
};

int main()
{
    string trainPath = "training/data/path";
    string trainTargetPath = "training/data/target/path";
    string testDataPath = "testing/data/path";
    string testTargetPath = "testing/data/target/path";
    
    // Training 
    
    cout << "Training Start..." << endl;
    int batchSize = 32;

    std::vector<NeuroVec<double>> data = ReadTxtFile(trainPath);
    vector<NeuroVec<NeuroVec<double>>> batchTrainingData = CreateMatrixGroup<double>(data, batchSize);

    vector<NeuroVec<double>> target = ReadTxtFile(trainTargetPath);
    vector<NeuroVec<double>> batchTargetData = CreateVectorGruop<double>(target, batchSize);
    
    cout << batchTrainingData.size() << "X" << batchTrainingData[0].len << "X" << batchTrainingData[0][0].len << endl;
    cout << batchTargetData.size() << "X" << batchTargetData[0].len << endl;

    nn neural;
    for(int epoch = 0; epoch < 500; epoch++)
    {
        double totalLoss = 0;
        for(int batch = 0; batch < batchTrainingData.size(); batch++)
        {
            NeuroVec<NeuroVec<double>> pred = neural.Forward(batchTrainingData[batch]);
            double loss = neural.loss(pred, batchTargetData[batch]);
            totalLoss += loss;
        }
        cout << "Epoch: " << epoch + 1 << "| Loss: " << totalLoss << endl;
    }

    // Testing 

    batchSize = 5;
    std::vector<NeuroVec<double>> testData = ReadTxtFile(testDataPath);
    vector<NeuroVec<NeuroVec<double>>> batchTestinggData = CreateMatrixGroup<double>(testData, batchSize);
    
    vector<NeuroVec<double>> targetTesting = ReadTxtFile(testTargetPath);
    vector<NeuroVec<double>> batchTestingTargetData = CreateVectorGruop<double>(targetTesting, batchSize);

    cout << batchTestinggData.size() << "X" << batchTrainingData[0].len << "X" << batchTestinggData[0][0].len << endl;
    cout << batchTestingTargetData.size() << "X" << batchTestingTargetData[0].len << endl;

    cout << "Testing Start..." << endl;
    for(int batch = 0; batch < 10; batch++)
    {
        NeuroVec<NeuroVec<double>> pred = neural.Forward(batchTestinggData[batch]);
        cout << "Prediction: " << endl;
        Print<double>(pred);
        cout << "Actual " << endl;
        Print<double>(batchTestingTargetData[batch]);
    }
    return 0;
}