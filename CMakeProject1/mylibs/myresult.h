#pragma once
#include "mylog.h"
#include <string>

#define DEFAULT_CODE 0
#define NORMAL_CODE 200
#define EXCEPTION_CODE 400
#define NULL_POINTER_EXCEPTION_CODE 401

#define DEFAULT_MESSAGE ""

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
    MyResult(Status status, int code, const char *message, T data, MyResult<T> *innerResult);

    bool IsException();
    MyResult<T>* GetInnerResult();
    MyResult<T>* Print();
    ~MyResult();

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

    void PrintRecursive();
};

template class MyResult<void *>;

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
        PrintLn("Result { Code: %d, Message: %s } ", this->_Code, this->_Message);
        return;
    }
    this->_InnerResult->Print();
    PrintLn("Result { Code: %d, Message: %s } ", this->_Code, this->_Message);
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
 * @brief 销毁该返回栈
 *
 * @tparam T
 */
template <class T>
MyResult<T>::~MyResult()
{
    if (this->_InnerResult == nullptr)
    {
        return;
    }
    delete this->_InnerResult;
}
