#include "RNN.hpp"
#include "CrossEntropyLossFunction.hpp"
#include "Softmax.hpp"

#include <fstream>
#include <unordered_map>

using namespace std;

vector<string> split(string str, char split)
{
    string temp = "";
    vector<string> res;
    for(int i = 0; i < str.size(); i++)
    {
        if(split == str[i])
        {
            res.push_back(temp);
            temp = "";
            continue;
        }
        temp += str[i];
    }
    if(temp != "")
        res.push_back(temp);
    return res;
}

vector<NeuroVec<NeuroVec<double>>> CreateTarget(vector<NeuroVec<NeuroVec<double>>> &input)
{
    vector<NeuroVec<NeuroVec<double>>> output;
    for(int i = 1; i < input.size(); i++)
    {
        NeuroVec<NeuroVec<double>> temp = CopyMatrix<double>(input[i]);
        output.push_back(temp);
    }
    return output;
}

vector<NeuroVec<NeuroVec<double>>> CreateSeq(int seqSize, int batchSize, int inputDim)
{
    vector<NeuroVec<NeuroVec<double>>> res;
    NeuroVec<NeuroVec<double>> temp = CreateMatrix<double>(batchSize, inputDim, 0);

    for(int i = 0; i < seqSize; i++)
    {      
        res.push_back(temp);
        temp = CreateMatrix<double>(batchSize, inputDim, 0);
    }
    return res;
}

class NextWordPrediction
{
public:
    NextWordPrediction(int inputDim, int outputDim) : rnn(inputDim, outputDim, 300), vocabSize(inputDim){}

    vector<NeuroVec<NeuroVec<double>>> CreateGroupData(int index, int batchSize)
    {
        int maxEle = 0;
        vector<vector<string>> sentences;

        for(int j = 0; j < batchSize; j++)
        {
            if(index+j >= data.size())
                break;
            vector<string> temp = split(data[index + j], ' ');  
            if(temp.size() > maxEle)
                maxEle = temp.size();
            sentences.push_back(temp);
        }

        vector<NeuroVec<NeuroVec<double>>> res = CreateSeq(maxEle, sentences.size(), vocabSize);
        for(int j = 0; j < sentences.size(); j++)
        {
            for(int k = 0; k < sentences[j].size(); k++)
            {
                res[k][j][words[sentences[j][k]]] = 1;
            }
        }

        return res;
    }

    void Train(int batch)
    {   
        int count = 0;
        cout << "start debuging" << endl;
        for(int epoch = 0; epoch < 1; epoch++)
        {
            cout << epoch << endl;
            while(data.size() > count)
            {
                cout << "first" << endl;
                vector<NeuroVec<NeuroVec<double>>> dataBatched = CreateGroupData(count, batch);
                vector<NeuroVec<NeuroVec<double>>> target = CreateTarget(dataBatched);

                cout << "sec" << endl;
                // Forward Prapogation
                vector<NeuroVec<NeuroVec<double>>> output = rnn.Forward(dataBatched);
                cout << "Run RNN" << endl;
                vector<NeuroVec<NeuroVec<double>>> prob = sf.Forward(output);

                // Loss
                cout << "softmax " << prob << endl;
                vector<NeuroVec<double>> loss = crLoss.Forward(prob, target);
                cout << "softmax End" << endl;
                int totalLoss = 0;
                for(int i = 0; i < loss.size(); i++)
                {
                    for(int j = 0; j < loss[i].len; j++)
                    {
                        totalLoss += loss[i][j];
                    }
                }
                cout << "Epoch: " << epoch + 1 << "Loss: " << totalLoss << endl;
                
                // BackWord Prapogation
                vector<NeuroVec<NeuroVec<double>>> grad = crLoss.Backward();
                sf.Backward(grad);

                cout << count << endl;
                count += batch;
            }
        }
    }
    
    void Predict()
    {

    }

    void MakeData(string pathData)
    {
        fstream newFile;
        string temp;
        vector<string> res;
        
        newFile.open(pathData, ios::in);
        if (!newFile.is_open())
        {
            std::cerr << "Error: Could not open file " << pathData << std::endl;
            exit(0);
        }
        string addString = "";
        while (getline(newFile, temp))
        {
            if(temp == "||")
            {
                res.push_back(addString);
                addString = "";
                continue;
            }
            if (temp != "")
                addString += (temp + " + ");
        }

        data = res;
        int count = 1;
        for(int i = 0; i < data.size(); i++)
        {
            vector<string> spliteSent = split(data[i], ' ');
            for(string word:spliteSent)
            {
                if(words.find(word) == words.end())
                {
                    words[word] = count;
                    count ++;
                }
            }
        }
    }

    void test()
    {
        int k = 0;
        for(auto key:words)
        {
            k+=1;
        }

        cout << k << endl;
    }

private:
    RNN rnn;
    vector<string> data;
    unordered_map<string, int> words;
    Sofmax sf;
    CrossEntropy crLoss;
    int vocabSize;
};

int main()
{
    // 42405
    NextWordPrediction model(42405, 42405);
    
    model.MakeData("C:\\Users\\shiva\\Desktop\\IISC\\code\\NeuroCpp\\NeuroVec-RNN\\dataset\\poem.txt");
    cout << "start Training..." << endl;
    model.Train(32);
    return 0;
}