#pragma once
#include "logs.hpp"
#include "thread.hpp"
#include "memory.hpp"
#include <string>
#include <vector>

namespace Kamanri
{
    namespace Utils
    {

        namespace Result$
        {
            constexpr int DEFAULT_CODE = 0;
            constexpr int NORMAL_CODE = 200;
            constexpr int GENERAL_EXCEPTION_CODE = 400;
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
        }

        // The `Result` class which provides capacity of returning exceptions.
        //
        // Note that:
        // - the parameter `data` only receive lvalue.
        // - need the default Constructor.
        // - need the Copy Constructor of type T.
        // - need the `operator=` of type T to initialze the result.
        // For the instance:
        // ```
        // Object() = default;
        // Object(Object& obj);
        // Object& operator=(Object& obj);
        // ```
        template <class T>
        class Result
        {
        public:
            enum class Status
            {
                NORMAL,
                EXCEPTION
            };
            // constructors
            Result();
            Result(Result<T>&& result) noexcept;
            explicit Result(T data);
            Result(Result<T>::Status status, int code, std::string const &message);
            Result(Status status, int code, std::string const &message, T data, P<Result<T>> innerResult);
            Result(Status status, int code, std::string const &message, P<Result<T>> innerResult, std::vector<Result$::StackTrace> &stackTrace);
            Result(Status status, int code, std::string const &message, T data, P<Result<T>> innerResult, std::vector<Result$::StackTrace> &stackTrace);

            bool IsException();
            // Result<T> *InnerResult();
            Result<T> *Print(bool is_print = true);
            Result<T> *PushToStack(Result$::StackTrace stackTrace);

            int Code() const;
            T &Data();
            T &operator*();

            template <class T2>
            Result<T2> As();
            template <class T2>
            Result<T2> As(T2 data);

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
            P<Result<T>> _InnerResult;
            // 调用堆栈
            std::vector<Result$::StackTrace> _StackTrace;

            void PrintOnce();

            void PrintRecursive();
        };

        using DefaultResult = Result<void *>;

        // template <class T>
        // using PResult = P<Result<T>>;

        template <class T>
        Result<T>::Result()
        {
            this->_Status = Result<T>::Status::NORMAL;
            this->_Code = Result$::DEFAULT_CODE;
            this->_Message = Result$::DEFAULT_MESSAGE;
            this->_InnerResult = nullptr;
        }

        template <class T>
        Result<T>::Result(Result<T>&& result) noexcept: 
        _Status(result._Status),
        _Code(result._Code),
        _Message(result._Message),
        _Data(result._Data)
        {
            this->_InnerResult.reset(result._InnerResult.release());
            this->_StackTrace.assign(result._StackTrace.begin(), _StackTrace.end());
        }

        template <class T>
        Result<T>::Result(T data)
        {
            this->_Status = Result<T>::Status::NORMAL;
            this->_Code = Result$::NORMAL_CODE;
            this->_Message = Result$::DEFAULT_MESSAGE;
            this->_Data = data;
            this->_InnerResult = nullptr;
        }

        template <class T>
        Result<T>::Result(
            Result<T>::Status status,
            int code,
            std::string const &message)
        {

            this->_Status = status;
            this->_Code = code;
            this->_Message = message;
            this->_InnerResult = nullptr;
        }

        template <class T>
        Result<T>::Result(
            Result<T>::Status status,
            int code,
            std::string const &message,
            T data,
            P<Result<T>> innerResult)
        {

            this->_Status = status;
            this->_Code = code;
            this->_Message = message;
            this->_Data = data;
            this->_InnerResult.reset(innerResult.release());
        }

        template <class T>
        Result<T>::Result(
            Status status,
            int code,
            std::string const &message,
            P<Result<T>> innerResult,
            std::vector<Result$::StackTrace> &stackTrace)
        {
            this->_Status = status;
            this->_Code = code;
            this->_Message = message;
            this->_InnerResult.reset(innerResult.release());
            this->_StackTrace.assign(stackTrace.begin(), stackTrace.end());
        }

        template <class T>
        Result<T>::Result(
            Status status,
            int code,
            std::string const &message,
            T data,
            P<Result<T>> innerResult,
            std::vector<Result$::StackTrace> &stackTrace)
        {
            this->_Status = status;
            this->_Code = code;
            this->_Message = message;
            this->_Data = data;
            this->_InnerResult.reset(innerResult.release());
            this->_StackTrace.assign(stackTrace.begin(), stackTrace.end());
        }

        template <class T>
        bool Result<T>::IsException()
        {
            return this->_Status == Status::EXCEPTION;
        }

        // template <class T>
        // Result<T> Result<T>::InnerResult()
        // {
        //     if (this->_InnerResult == nullptr)
        //     {
        //         return Result<T>(
        //             Status::EXCEPTION,
        //             Result$::NULL_POINTER_EXCEPTION_CODE,
        //             "The Inner Result Is Null",
        //             nullptr,
        //             nullptr);
        //     }
        //     return this->_InnerResult;
        // }

        template <class T>
        void Result<T>::PrintOnce()
        {
            PrintLn("Result %d: %s", this->_Code, this->_Message.c_str());
            std::vector<Result$::StackTrace> copiedStackTrace(this->_StackTrace);
            while (!copiedStackTrace.empty())
            {
                auto popStackIterator = copiedStackTrace.begin();
                popStackIterator->Print();
                copiedStackTrace.erase(popStackIterator);
            }
        }

        template <class T>
        void Result<T>::PrintRecursive()
        {
            if (this->_InnerResult == nullptr)
            {
                if (this->IsException())
                {
                    Log::Error("ResultError", "An Exception In Result Occurred Caused By: ");
                    Log::Error(
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
        Result<T> *Result<T>::Print(bool is_print)
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
        Result<T> *Result<T>::PushToStack(Result$::StackTrace stackTrace)
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
        T &Result<T>::Data()
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
        T &Result<T>::operator*()
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
        int Result<T>::Code() const
        {
            return _Code;
        }

        template <class T>
        template <class T2>
        Result<T2> Result<T>::As()
        {
            if (this->_InnerResult == nullptr)
            {
                return Result<T2>(
                    this->_Status == Result<T>::EXCEPTION ? Result<T2>::EXCEPTION : Result<T2>::NORMAL,
                    this->_Code,
                    this->_Message,
                    P<Result<T2>>(nullptr),
                    this->_StackTrace);
            }
            return Result<T2>(
                this->_Status == Result<T>::EXCEPTION ? Result<T2>::EXCEPTION : Result<T2>::NORMAL,
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
        Result<T2> Result<T>::As(T2 data)
        {
            if (this->_InnerResult == nullptr)
            {
                return Result<T2>(
                    this->_Status == Result<T>::EXCEPTION ? Result<T2>::EXCEPTION : Result<T2>::NORMAL,
                    this->_Code,
                    this->_Message,
                    data,
                    P<Result<T2>>(nullptr),
                    this->_StackTrace);
            }
            return Result<T2>(
                this->_Status == Result<T>::EXCEPTION ? Result<T2>::EXCEPTION : Result<T2>::NORMAL,
                this->_Code,
                this->_Message,
                data,
                this->_InnerResult->As(data),
                this->_StackTrace);
        }

    }
}

///////////////////////////////////////////////////////////////////////////////////////

#define TRY_FOR_TYPE(T, result)                                                      \
    auto _res_ = result;                                                             \
    if (_res_.IsException())                                                        \
    {                                                                                \
        _res_.PushToStack(Kamanri::Utils::Result$::StackTrace(__FILE__, __LINE__)); \
        return _res_.As<T>();                                                       \
    }

#define TRY(result)                                                                  \
    auto _res_ = result;                                                             \
    if (_res_.IsException())                                                        \
    {                                                                                \
        _res_.PushToStack(Kamanri::Utils::Result$::StackTrace(__FILE__, __LINE__)); \
        return _res_;                                                                \
    }

#define DEFAULT_RESULT Kamanri::Utils::Result<void *>()

#define DEFAULT_RESULT_EXCEPTION(code, message) Kamanri::Utils::Result<void *>(Kamanri::Utils::Result<void *>::Status::EXCEPTION, code, message)

#define RESULT_EXCEPTION(T, code, message) Kamanri::Utils::Result<T>(Kamanri::Utils::Result<T>::Status::EXCEPTION, code, message)

#define CHECK_MEMORY_FOR_DEFAULT_RESULT(p, log_name, code)                           \
    CHECK_MEMORY_IS_ALLOCATED(p, log_name,                                           \
                              Kamanri::Utils::Result<void *>(                        \
                                  Kamanri::Utils::Result<void *>::Status::EXCEPTION, \
                                  code,                                              \
                                  "The memory is not initialized"))

#define CHECK_MEMORY_FOR_RESULT(T, p, log_name, code)                           \
    CHECK_MEMORY_IS_ALLOCATED(p, log_name,                                      \
                              Kamanri::Utils::Result<T>(                        \
                                  Kamanri::Utils::Result<T>::Status::EXCEPTION, \
                                  code,                                         \
                                  "The memory is not initialized"))
