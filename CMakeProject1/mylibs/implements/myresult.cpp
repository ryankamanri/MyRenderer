#include "../myresult.h"
#include <fstream>

StackTrace::StackTrace(const char* file, int line, int lineCount): _File(file), _Line(line), _LineCount(lineCount)
{
    // auto get the statement
    _Statement = nullptr;

    // read the source file
    std::ifstream fs(file);

    std::string str;
    std::string* statement = new std::string();
    int index = 1;
    while (index < line && getline(fs, str))
    {
        index++;
    }
    while (index < line + lineCount && getline(fs, str))
    {
        *statement += "\n\t\t";
        *statement += str;
        index++;
    }
    
    fs.close();
    if(index < line)
    {
        Log::Error("MyResultStackTraceException", "File %s Lines Count %d < %d (The Given)", file, index, line);
        Log::Warn("MyResultStackTraceException", 
        "It Might Due To Your Code Source File '%s' Has Been Occupied By Another Process, Try To Close All Process Which Keep Occupying It, And The Following StackTrace Might Be In A Mess, Do NOT Care.", 
        file);
        return;
    }
    
    _Statement = statement->c_str();
    
}

void StackTrace::Print() 
{
    PrintLn("\tat %s, Line %d - %d", this->_File, this->_Line, this->_Line + this->_LineCount);
    if(_Statement != nullptr) {
        PrintLn("\t\t%s", _Statement);
    }
    
}