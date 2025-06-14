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

    vector<vector<NeuroVec<NeuroVec<double>>>> CreateGroupData(int batchSize)
    {
        vector<vector<NeuroVec<NeuroVec<double>>>> res;
        vector<NeuroVec<NeuroVec<double>>> temp;
        for(int i = 0; i < data.size(); i+=batchSize)
        {
            int max = -1;
            vector<vector<string>> sentences;
            for(int j = 0; j < batchSize; j++)
            {
                if(i+j >= data.size())
                    break;
                vector<string> temp = split(data[i + j], ' ');
                if(temp.size() > max)
                    max = temp.size();
                sentences.push_back(temp);
            }
            temp = CreateSeq(max, sentences.size(), vocabSize);
            for(int j = 0; j < sentences.size(); j++)
            {
                for(int k = 0; k < sentences[j].size(); k++)
                {
                    temp[k][j][words[sentences[j][k]]] = 1;
                }
            }
            res.push_back(temp);
        }
        return res;
    }

    void Train(int batch)
    {
        
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
            if(i < 10)
                cout << data[i] << endl;
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
    int vocabSize;
};

int main()
{
    NextWordPrediction model(42405, 42405);
    
    model.MakeData("C:\\Users\\shiva\\Desktop\\IISC\\code\\NeuroCpp\\NeuroVec-RNN\\dataset\\poem.txt");
    model.test();
    return 0;
}