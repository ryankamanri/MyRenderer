#pragma once
#include "mylog.h"
#include "mythread.h"
#include <string>


#define DEFAULT_CODE 0
#define NORMAL_CODE 200
#define EXCEPTION_CODE 400
#define NULL_POINTER_EXCEPTION_CODE 401

#define DEFAULT_MESSAGE ""



/**
 * @brief 堆栈跟踪
 * 
 */
typedef struct StackTrace
{
    StackTrace(const char* file, int line, int lineCount);
    /* data */
    const char* _File;
    int _Line;
    int _LineCount;
    const char* _Statement;
    void Print();
} StackTrace;


StackTrace::StackTrace(const char* file, int line, int lineCount = 1): _File(file), _Line(line), _LineCount(lineCount)
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
        Log<const char*, int, int>::Error("MyResultStackTraceException", "File %s Lines Count %d < %d (The Given)", file, index, line);
        Log<const char*>::Warn("MyResultStackTraceException", "It Might Due To Your Code Source File '%s' Has Been Occupied By Another Process, Try To Close All Process Which Keep Occupying It.", file);
        Log<>::Warn("MyResultStackTraceException", "And The Following StackTrace Might Be In A Mess, Do NOT Care.");
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


template <class T>
class MyResult
{
public:
    enum Status
    {
        NORMAL,
        EXCEPTION
    };
    // constructors
    MyResult();
    explicit MyResult(T data);
    MyResult(Status status, int code, const char *message, T data, MyResult<T> *innerResult);
    MyResult(Status status, int code, const char *message, T data, MyResult<T> *innerResult, std::vector<StackTrace> stackTrace);
    ~MyResult();

    bool IsException();
    MyResult<T>* GetInnerResult();
    MyResult<T>* Print();
    MyResult<T>* PushToStack(StackTrace stackTrace);

    void Dispose();
    T Data();

    template <class T2>
    MyResult<T2>* As(T2 data);


private:
    // 返回状态: NORMAL为正常返回, EXCEPTION为异常返回
    Status _Status;
    // 状态码, 可自定义
    int _Code;
    // 消息, 可自定义
    const char *_Message;
    // 返回值
    T _Data;
    // 内部返回或内部异常
    MyResult<T> *_InnerResult;
    // 调用堆栈
    std::vector<StackTrace> _StackTrace;

    void PrintOnce();

    void PrintRecursive();
};



template <class T>
MyResult<T>::MyResult()
{
    this->_Status = MyResult<T>::Status::NORMAL;
    this->_Code = DEFAULT_CODE;
    this->_Message = DEFAULT_MESSAGE;
    this->_Data = nullptr;
    this->_InnerResult = nullptr;
}

template <class T>
MyResult<T>::MyResult(T data)
{
    this->_Status = MyResult<T>::Status::NORMAL;
    this->_Code = NORMAL_CODE;
    this->_Message = DEFAULT_MESSAGE;
    this->_Data = data;
    this->_InnerResult = nullptr;
}


template <class T>
MyResult<T>::MyResult(
    MyResult<T>::Status status,
    int code,
    const char *message,
    T data,
    MyResult<T> *innerResult)
{

    this->_Status = status;
    this->_Code = code;
    this->_Message = message;
    this->_Data = data;
    this->_InnerResult = innerResult;
}

template <class T>
MyResult<T>::MyResult(
    Status status, 
    int code, 
    const char *message, 
    T data, 
    MyResult<T> *innerResult, 
    std::vector<StackTrace> stackTrace)
{
    this->_Status = status;
    this->_Code = code;
    this->_Message = message;
    this->_Data = data;
    this->_InnerResult = innerResult;
    this->_StackTrace.assign(stackTrace.begin(), stackTrace.end());
}

/**
 * @brief 销毁该返回栈
 *
 * @tparam T
 */
template <class T>
MyResult<T>::~MyResult()
{

}

template <class T>
bool MyResult<T>::IsException()
{
    return this->_Status == Status::EXCEPTION;
}

template <class T>
MyResult<T>* MyResult<T>::GetInnerResult()
{
    if (this->_InnerResult == nullptr)
    {
        return new MyResult<void *>(
            Status::EXCEPTION,
            NULL_POINTER_EXCEPTION_CODE,
            "The Inner Result Is Null",
            nullptr,
            nullptr);
    }
    return this->_InnerResult;
}

template <class T>
void MyResult<T>::PrintOnce()
{
    PrintLn("Result %d: %s", this->_Code, this->_Message);
    std::vector<StackTrace> copiedStackTrace(this->_StackTrace);
    while (!copiedStackTrace.empty())
    {
        std::vector<StackTrace>::iterator popStackIterator = copiedStackTrace.begin();
        popStackIterator.base()->Print();
        copiedStackTrace.erase(popStackIterator);
    }
    
}

template <class T>
void MyResult<T>::PrintRecursive()
{
    if (this->_InnerResult == nullptr)
    {
        if (this->IsException())
        {
            Log<>::Error("ResultError", "An Exception In Result Occurred Caused By: ");
            Log<>::Error(
                std::to_string(this->_Code).c_str(),
                this->_Message);
        }
        this->PrintOnce();
        return;
    }
    this->_InnerResult->PrintRecursive();
    this->PrintOnce();
}

/**
 * @brief 打印调用栈
 *
 * @tparam T
 * @return MyResult<T>
 */
template <class T>
MyResult<T>* MyResult<T>::Print()
{
    this->PrintRecursive();
    return this;
}


/**
 * @brief 压入调用栈
 * 
 * @tparam T 
 * @param stackTrace 
 * @return MyResult<T>* 
 */
template <class T>
MyResult<T>* MyResult<T>::PushToStack(StackTrace stackTrace)
{
    this->_StackTrace.push_back(stackTrace);
    return this;
}


/**
 * @brief Dispose this result.
 * 
 * @tparam T 
 */
template<class T>
void MyResult<T>::Dispose()
{
    delete this;
}

/**
 * @brief Get The Return Data And Dispose result
 * 
 * @tparam T 
 * @return T 
 */
template<class T>
T MyResult<T>::Data()
{
    T data = this->_Data;
    if(this->IsException()) {
        this->Print();
    }
    delete this;
    return data;
}

template <class T>
template <class T2>
MyResult<T2> *MyResult<T>::As(T2 data)
{
    if (this->_InnerResult == nullptr)
    {
        return new MyResult<T2>(
            this->_Status == MyResult<T>::EXCEPTION ? MyResult<T2>::EXCEPTION : MyResult<T2>::NORMAL,
            this->_Code,
            this->_Message,
            data,
            nullptr,
            this->_StackTrace);
    }
    return new MyResult<T2>(
        this->_Status == MyResult<T>::EXCEPTION ? MyResult<T2>::EXCEPTION : MyResult<T2>::NORMAL,
        this->_Code,
        this->_Message,
        data,
        this->_InnerResult->As(data),
        this->_StackTrace);
}
