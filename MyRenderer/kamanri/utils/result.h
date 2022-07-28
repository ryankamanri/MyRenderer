#pragma once
#include "logs.h"
#include "thread.h"
#include "memory.h"
#include <string>
#include <vector>


namespace Kamanri
{
    namespace Utils
    {
        namespace Result
        {

            constexpr int DEFAULT_CODE = 0;
            constexpr int NORMAL_CODE = 200;
            constexpr int EXCEPTION_CODE = 400;
            constexpr int NULL_POINTER_EXCEPTION_CODE = 401;

            constexpr const char *DEFAULT_MESSAGE = "";

            /**
             * @brief 堆栈跟踪
             *
             */
            using StackTrace = struct StackTrace
            {
                StackTrace(std::string const &file, int line, int lineCount = 1);
                /* data */
                std::string _File;
                int _Line;
                int _LineCount;
                std::string _Statement;
                void Print() const;
            };

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
                MyResult(MyResult<T>::Status status, int code, std::string const &message);
                MyResult(Status status, int code, std::string const &message, T data, Memory::P<MyResult<T>> innerResult);
                MyResult(Status status, int code, std::string const &message, Memory::P<MyResult<T>> innerResult, std::vector<StackTrace> &stackTrace);
                MyResult(Status status, int code, std::string const &message, T data, Memory::P<MyResult<T>> innerResult, std::vector<StackTrace> &stackTrace);

                bool IsException();
                MyResult<T> *InnerResult();
                MyResult<T> *Print(bool is_print = true);
                MyResult<T> *PushToStack(StackTrace stackTrace);

                int Code() const;
                T &Data();
                T &operator*();

                template <class T2>
                Memory::P<MyResult<T2>> As();
                template <class T2>
                Memory::P<MyResult<T2>> As(T2 data);

            private:
                // 返回状态: NORMAL为正常返回, EXCEPTION为异常返回
                Status _Status;
                // 状态码, 可自定义
                int _Code;
                // 消息, 可自定义
                std::string _Message;
                // 返回值
                T _Data;
                // 内部返回或内部异常
                Memory::P<MyResult<T>> _InnerResult;
                // 调用堆栈
                std::vector<StackTrace> _StackTrace;

                void PrintOnce();

                void PrintRecursive();
            };

            using DefaultResult = Memory::P<MyResult<void *>>;

            template<class T>
            using PMyResult = Memory::P<MyResult<T>>;

            template <class T>
            MyResult<T>::MyResult()
            {
                this->_Status = MyResult<T>::Status::NORMAL;
                this->_Code = DEFAULT_CODE;
                this->_Message = DEFAULT_MESSAGE;
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
                std::string const &message)
            {

                this->_Status = status;
                this->_Code = code;
                this->_Message = message;
                this->_InnerResult = nullptr;
            }

            template <class T>
            MyResult<T>::MyResult(
                MyResult<T>::Status status,
                int code,
                std::string const &message,
                T data,
                Memory::P<MyResult<T>> innerResult)
            {

                this->_Status = status;
                this->_Code = code;
                this->_Message = message;
                this->_Data = data;
                this->_InnerResult.reset(innerResult.release());
            }

            template <class T>
            MyResult<T>::MyResult(
                Status status,
                int code,
                std::string const &message,
                Memory::P<MyResult<T>> innerResult,
                std::vector<StackTrace> &stackTrace)
            {
                this->_Status = status;
                this->_Code = code;
                this->_Message = message;
                this->_InnerResult.reset(innerResult.release());
                this->_StackTrace.assign(stackTrace.begin(), stackTrace.end());
            }

            template <class T>
            MyResult<T>::MyResult(
                Status status,
                int code,
                std::string const &message,
                T data,
                Memory::P<MyResult<T>> innerResult,
                std::vector<StackTrace> &stackTrace)
            {
                this->_Status = status;
                this->_Code = code;
                this->_Message = message;
                this->_Data = data;
                this->_InnerResult.reset(innerResult.release());
                this->_StackTrace.assign(stackTrace.begin(), stackTrace.end());
            }

            template <class T>
            bool MyResult<T>::IsException()
            {
                return this->_Status == Status::EXCEPTION;
            }

            template <class T>
            MyResult<T> *MyResult<T>::InnerResult()
            {
                if (this->_InnerResult == nullptr)
                {
                    return Memory::New<MyResult<T>>(
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
                Logs::PrintLn("Result %d: %s", this->_Code, this->_Message.c_str());
                std::vector<StackTrace> copiedStackTrace(this->_StackTrace);
                while (!copiedStackTrace.empty())
                {
                    auto popStackIterator = copiedStackTrace.begin();
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
                        Logs::Log::Error("ResultError", "An Exception In Result Occurred Caused By: ");
                        Logs::Log::Error(
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
            MyResult<T> *MyResult<T>::Print(bool is_print)
            {
                if (is_print)
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
            MyResult<T> *MyResult<T>::PushToStack(StackTrace stackTrace)
            {
                this->_StackTrace.push_back(stackTrace);
                return this;
            }

            /**
             * @brief Get The Return Data
             *
             * @tparam T
             * @return T&
             */
            template <class T>
            T &MyResult<T>::Data()
            {
                if (this->IsException())
                {
                    this->Print();
                }
                return this->_Data;
            }

            /**
             * @brief Get The Return Data
             *
             * @tparam T
             * @return T&
             */
            template <class T>
            T &MyResult<T>::operator*()
            {
                if (this->IsException())
                {
                    this->Print();
                }
                return this->_Data;
            }

            /**
             * @brief Get the code of this result.
             *
             * @tparam T
             * @return int
             */
            template <class T>
            int MyResult<T>::Code() const
            {
                return _Code;
            }

            template <class T>
            template <class T2>
            Memory::P<MyResult<T2>> MyResult<T>::As()
            {
                if (this->_InnerResult == nullptr)
                {
                    return Memory::New<MyResult<T2>>(
                        this->_Status == MyResult<T>::EXCEPTION ? MyResult<T2>::EXCEPTION : MyResult<T2>::NORMAL,
                        this->_Code,
                        this->_Message,
                        Memory::P<MyResult<T2>>(nullptr),
                        this->_StackTrace);
                }
                return Memory::New<MyResult<T2>>(
                    this->_Status == MyResult<T>::EXCEPTION ? MyResult<T2>::EXCEPTION : MyResult<T2>::NORMAL,
                    this->_Code,
                    this->_Message,
                    this->_InnerResult->As<T2>(),
                    this->_StackTrace);
            }

            /**
             * @brief Change this result to another type(T2).
             *
             * @tparam T
             * @tparam T2
             * @param data
             * @return MyResult<T2>*
             */
            template <class T>
            template <class T2>
            Memory::P<MyResult<T2>> MyResult<T>::As(T2 data)
            {
                if (this->_InnerResult == nullptr)
                {
                    return Memory::New<MyResult<T2>>(
                        this->_Status == MyResult<T>::EXCEPTION ? MyResult<T2>::EXCEPTION : MyResult<T2>::NORMAL,
                        this->_Code,
                        this->_Message,
                        data,
                        Memory::P<MyResult<T2>>(nullptr),
                        this->_StackTrace);
                }
                return Memory::New<MyResult<T2>>(
                    this->_Status == MyResult<T>::EXCEPTION ? MyResult<T2>::EXCEPTION : MyResult<T2>::NORMAL,
                    this->_Code,
                    this->_Message,
                    data,
                    this->_InnerResult->As(data),
                    this->_StackTrace);
            }


        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////

#define SOURCE_FILE(location) constexpr const char *SOURCE_FILE_LOCATION = location

#define THROW_EXCEPTION_FOR_TYPE(T, result, line)                    \
    if (result->IsException())                                       \
    {                                                                \
        result->PushToStack(StackTrace(SOURCE_FILE_LOCATION, line)); \
        return result->As<T>();                                      \
    }

#define THROW_EXCEPTION(result, line)                                \
    if (result->IsException())                                       \
    {                                                                \
        result->PushToStack(StackTrace(SOURCE_FILE_LOCATION, line)); \
        return result;                                               \
    }

            

#define DEFAULT_RESULT New<MyResult<void *>>()

#define DEFAULT_RESULT_EXCEPTION(code, message) New<MyResult<void *>>(MyResult<void *>::EXCEPTION, code, message)

#define RESULT_EXCEPTION(T, code, message) New<MyResult<T>>(MyResult<T>::EXCEPTION, code, message)