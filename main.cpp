#include "NeuroVec.hpp"

#include <vector>
#include <utility>
#include <string>
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
            // cout << temp << endl;
            if(temp == "")
                continue;
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

// Extract poem and word vector
pair<vector<string>, unordered_map<string, NeuroVec<double>>> MakeData(string pathData, string wordPath)
{
    fstream newFile, wordFile;
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


    wordFile.open(wordPath, ios::in);
    if (!wordFile.is_open())
    {
        std::cerr << "Error: Could not open file " << pathData << std::endl;
        exit(0);
    }
    unordered_map<string, NeuroVec<double>> wordEmb;
 
    while (getline(wordFile, temp))
    {
        vector<string> emb = split(temp, ' ');
        if(wordEmb.find(emb[0]) == wordEmb.end())
        {            
            wordEmb[emb[0]] = CreateVector<double>(emb.size() - 1, 0);
            for(int i = 1; i < emb.size(); i++)
            {
                wordEmb[emb[0]][i - 1] = std::stod(emb[i]);                
            }
        }
    }
        
    pair<vector<string>, unordered_map<string, NeuroVec<double>>> returnData;
    returnData.first = res;
    returnData.second = wordEmb;
    return returnData;
}

class PoemPredict
{
public:
    PoemPredict(vector<string> _poemData, unordered_map<string, NeuroVec<double>> _wordEmb): poemData(_poemData), wordEmb(_wordEmb)
    {}

    void Train()
    {

    }

private:
    vector<string> poemData;
    unordered_map<string, NeuroVec<double>> wordEmb;
};

PoemPredict CreateModel(string path)
{

}

int main()
{
    pair<vector<string>, unordered_map<string, NeuroVec<double>>> data = MakeData("c:\\Users\\shiva\\Desktop\\IISC\\code\\NeuroCpp\\NeuroVec-RNN\\glove.6B\\poem.txt", "C:\\Users\\shiva\\Desktop\\IISC\\code\\NeuroCpp\\NeuroVec-RNN\\glove.6B\\GetRelatedData.txt");    
    cout << data.first.size() << endl;
    Print<double>(data.second["the"]);
    PoemPredict model(data.first, data.second);
    return 0;
}