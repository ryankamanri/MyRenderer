#include <string>
#include <vector>
#include "../../utils/string.hpp"


std::vector<std::string> Kamanri::Utils::String::Split(const std::string &str, const std::string &delim, bool is_remove_empty)
{ //将分割后的子字符串存储在vector中

    std::vector<std::string> res;
    if ("" == str)
        return res;

    std::string strs = str + delim; //*****扩展字符串以方便检索最后一个分隔出的字符串
    size_t pos;
    size_t size = strs.size();

    for (int i = 0; i < size; ++i)
    {
        pos = strs.find(delim, i); // pos为分隔符第一次出现的位置，从i到pos之前的字符串是分隔出来的字符串
        if (pos < size)
        {                                            //如果查找到，如果没有查找到分隔符，pos为string::npos
            std::string s = strs.substr(i, pos - i); //*****从i开始长度为pos-i的子字符串
            if(is_remove_empty && s.length() == 0) continue;
            res.push_back(s);                        //两个连续空格之间切割出的字符串为空字符串，这里没有判断s是否为空，所以最后的结果中有空字符的输出，
            i = pos + delim.size() - 1;
        }
    }
    return res;
}
