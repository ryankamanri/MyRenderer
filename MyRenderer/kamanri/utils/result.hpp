#pragma once
#ifndef SWIG
#include "log.hpp"
#include "memory.hpp"
#include <string>
#include <vector>
#endif

namespace Kamanri
{
	namespace Utils
	{

		namespace Result$
		{
			constexpr int DEFAULT_CODE = 0;
			constexpr int NORM_CODE = 200;
			constexpr int GENERAL_EXCEPTION_CODE = 400;
			constexpr int NULL_POINTER_EXCEPTION_CODE = 401;

			constexpr const char *DEFAULT_MESSAGE = "";

			/**
			 * @brief 堆栈跟踪
			 *
			 */
			struct StackTrace
			{
				StackTrace(std::string const &file, int line, int lineCount = 1);
				/* data */
				std::string _file;
				int _line;
				int _line_count;
				std::string _statement;
				void Print() const;
			};

			/**
			 * @brief Result status
			 *
			 */
			enum class Status
			{
				NORM,
				EXCEPTION
			};
		}

		// The `Result` class which provides capacity of returning exceptions.
		//
		// Note that:
		// - the parameter `data` only receive lvalue.
		// - need the default Constructor.
		// - need the Copy Constructor of type T.
		// - need the `operator=` of type T to initialize the result.
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
			// constructors
			Result();
			Result(Result<T> const& result);
			Result(Result<T> &&result);
			explicit Result(T data);
			Result(Result$::Status status, int code, std::string const &message);
			Result(Result$::Status status, int code, std::string const &message, T data, Kamanri::Utils::P<Result<T>>& innerResult);
			Result(Result$::Status status, int code, std::string const &message, Kamanri::Utils::P<Result<T>>& innerResult, std::vector<Result$::StackTrace> &stackTrace);
			Result(Result$::Status status, int code, std::string const &message, T data, Kamanri::Utils::P<Result<T>>& innerResult, std::vector<Result$::StackTrace> &stackTrace);

			Result<T>& operator=(Result<T> const& result);
			Result<T>& operator=(Result<T> &&result);

			bool IsException();
			// Result<T> *InnerResult();
			Result<T>& Print(bool is_print = true);
			Result<T>& PushToStack(Result$::StackTrace stackTrace);

			int Code() const;
			T &Data();
			T &operator*();

			template <class T2>
			Result<T2> As();
			template <class T2>
			Result<T2> As(T2 data);

		private:
			// 返回状态: NORM为正常返回, EXCEPTION为异常返回
			Result$::Status _status;
			// 状态码, 可自定义
			int _code;
			// 消息, 可自定义
			std::string _message;
			// 返回值
			T _data;
			// 内部返回或内部异常
			Kamanri::Utils::P<Result<T>> _inner_result;
			// 调用堆栈
			std::vector<Result$::StackTrace> _stacktrace;

			void PrintOnce();

			void PrintRecursive();
		};

		using DefaultResult = Result<void *>;

		// template <class T>
		// using PResult = P<Result<T>>;

		template <class T>
		Result<T>::Result(): _status(Result$::Status::NORM) {}

		template <class T>
		Result<T>::Result(Result<T> const& result) :
			_status(result._status),
			_code(result._code),
			_message(result._message),
			_data(result._data)
		{
			this->_inner_result = Copy(result._inner_result.get());
		}

		template <class T>
		Result<T>::Result(Result<T>&& result) :
			_status(result._status),
			_code(result._code),
			_message(result._message),
			_data(std::move(result._data))
		{
			this->_inner_result.reset(result._inner_result.release());
			this->_stacktrace.assign(result._stacktrace.begin(), _stacktrace.end());
		}

		template <class T>
		Result<T>::Result(T data)
		{
			this->_status = Result$::Status::NORM;
			this->_code = Result$::NORM_CODE;
			this->_message = Result$::DEFAULT_MESSAGE;
			this->_data = data;
			this->_inner_result = nullptr;
		}

		template <class T>
		Result<T>::Result(
			Result$::Status status,
			int code,
			std::string const &message)
		{

			this->_status = status;
			this->_code = code;
			this->_message = message;
			this->_inner_result = nullptr;
		}

		template <class T>
		Result<T>::Result(
			Result$::Status status,
			int code,
			std::string const &message,
			T data,
			P<Result<T>>& innerResult)
		{

			this->_status = status;
			this->_code = code;
			this->_message = message;
			this->_data = data;
			this->_inner_result.reset(innerResult.release());
		}

		template <class T>
		Result<T>::Result(
			Result$::Status status,
			int code,
			std::string const &message,
			P<Result<T>>& innerResult,
			std::vector<Result$::StackTrace> &stackTrace)
		{
			this->_status = status;
			this->_code = code;
			this->_message = message;
			this->_inner_result.reset(innerResult.release());
			if (this->_stacktrace.size()) this->_stacktrace.assign(stackTrace.begin(), stackTrace.end());
		}

		template <class T>
		Result<T>::Result(
			Result$::Status status,
			int code,
			std::string const &message,
			T data,
			P<Result<T>>& innerResult,
			std::vector<Result$::StackTrace> &stackTrace)
		{
			this->_status = status;
			this->_code = code;
			this->_message = message;
			this->_data = data;
			this->_inner_result.reset(innerResult.release());
			if (this->_stacktrace.size()) this->_stacktrace.assign(stackTrace.begin(), stackTrace.end());
		}

		template <class T>
		Result<T>& Result<T>::operator=(Result<T> const& result)
		{
			_status = result._status;
			_code = result._code;
			_message = result._message;
			_data = result._data;
			this->_inner_result = Copy(result._inner_result.get());
			this->_stacktrace = result._stacktrace;
			return *this;
		}

		template <class T>
		Result<T>& Result<T>::operator=(Result<T> &&result)
		{
			_status = result._status;
			_code = result._code;
			_message = std::move(result._message);
			_data = std::move(result._data);
			this->_inner_result.reset(result._inner_result.release());
			if(this->_stacktrace.size()) this->_stacktrace.assign(result._stacktrace.begin(), _stacktrace.end());
			return *this;
		}

		template <class T>
		bool Result<T>::IsException()
		{
			return this->_status == Result$::Status::EXCEPTION;
		}

		// template <class T>
		// Result<T> Result<T>::InnerResult()
		// {
		//     if (this->_inner_result == nullptr)
		//     {
		//         return Result<T>(
		//             Status::EXCEPTION,
		//             Result$::NULL_POINTER_EXCEPTION_CODE,
		//             "The Inner Result Is Null",
		//             nullptr,
		//             nullptr);
		//     }
		//     return this->_inner_result;
		// }

		template <class T>
		void Result<T>::PrintOnce()
		{
			PrintLn("Result %d: %s", this->_code, this->_message.c_str());
			std::vector<Result$::StackTrace> copiedStackTrace(this->_stacktrace);
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
			if (this->_inner_result == nullptr)
			{
				if (this->IsException())
				{
					Log::Error("ResultError", "An Exception In Result Occurred Caused By: ");
					Log::Error(
						std::to_string(this->_code).c_str(),
						this->_message);
				}
				this->PrintOnce();
				return;
			}
			this->_inner_result->PrintRecursive();
			this->PrintOnce();
		}

		/**
		 * @brief 打印调用栈
		 *
		 * @tparam T
		 * @return MyResult<T>
		 */
		template <class T>
		Result<T>& Result<T>::Print(bool is_print)
		{
			if (is_print)
				this->PrintRecursive();
			return *this;
		}

		/**
		 * @brief 压入调用栈
		 *
		 * @tparam T
		 * @param stackTrace
		 * @return MyResult<T>*
		 */
		template <class T>
		Result<T>& Result<T>::PushToStack(Result$::StackTrace stackTrace)
		{
			this->_stacktrace.push_back(stackTrace);
			return *this;
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
			return this->_data;
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
			return this->_data;
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
			return _code;
		}

		template <class T>
		template <class T2>
		Result<T2> Result<T>::As()
		{
			if (this->_inner_result == nullptr)
			{
				return Result<T2>(
					this->_status,
					this->_code,
					this->_message,
					P<Result<T2>>(nullptr),
					this->_stacktrace);
			}
			return Result<T2>(
				this->_status,
				this->_code,
				this->_message,
				New<Result<T2>>(this->_inner_result->As<T2>()),
				this->_stacktrace);
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
			if (this->_inner_result == nullptr)
			{
				return Result<T2>(
					this->_status,
					this->_code,
					this->_message,
					data,
					P<Result<T2>>(nullptr),
					this->_stacktrace);
			}
			return Result<T2>(
				this->_status,
				this->_code,
				this->_message,
				data,
				New<Result<T2>>(this->_inner_result->As(data)),
				this->_stacktrace);
		}

	}
}

///////////////////////////////////////////////////////////////////////////////////////

// Unique Name Generator
#define ___CAT___(a, b) a##b
#define __CAT__(a, b) ___CAT___(a, b)
#define _UNIQUE_NAME_(prefix) __CAT__(prefix, __LINE__)
#define _UNIQUE_RES_ _UNIQUE_NAME_(_res_)

#define ASSERT(result)                                                                     \
	auto _UNIQUE_RES_ = result;                                                            \
	if (_UNIQUE_RES_.IsException())                                                        \
	{                                                                                      \
		_UNIQUE_RES_.PushToStack(Kamanri::Utils::Result$::StackTrace(__FILE__, __LINE__)); \
		return _UNIQUE_RES_;                                                               \
	}

// Note: The type should have a move constructor.
#define TRY(result) std::move(({ASSERT(result); std::move(_UNIQUE_RES_); }).Data())

#define ASSERT_FOR_TYPE(T, result)                                                         \
	auto _UNIQUE_RES_ = result;                                                            \
	if (_UNIQUE_RES_.IsException())                                                        \
	{                                                                                      \
		_UNIQUE_RES_.PushToStack(Kamanri::Utils::Result$::StackTrace(__FILE__, __LINE__)); \
		return _UNIQUE_RES_.As<T>();                                                       \
	}

// Note: The type should have a move constructor.
#define TRY_FOR_TYPE(T, result) std::move(({ ASSERT_FOR_TYPE(T, result); std::move(_UNIQUE_RES_); }).Data())

#define ASSERT_FOR_DEFAULT(result) ASSERT_FOR_TYPE(void *, result)

#define TRY_FOR_DEFAULT(result) TRY_FOR_TYPE(void *, result)

#define DEFAULT_RESULT Kamanri::Utils::DefaultResult()

#define DEFAULT_RESULT_EXCEPTION(code, message) Kamanri::Utils::DefaultResult(Kamanri::Utils::Result$::Status::EXCEPTION, code, message)

#define RESULT_EXCEPTION(T, code, message) Kamanri::Utils::Result<T>(Kamanri::Utils::Result$::Status::EXCEPTION, code, message)

#define CHECK_MEMORY_FOR_DEFAULT_RESULT(p, log_name, code)                    \
	CHECK_MEMORY_IS_ALLOCATED(p, log_name,                                    \
							  Kamanri::Utils::DefaultResult(                 \
								  Kamanri::Utils::Result$::Status::EXCEPTION, \
								  code,                                       \
								  "The memory is not initialized"))

#define CHECK_MEMORY_FOR_RESULT(T, p, log_name, code)                         \
	CHECK_MEMORY_IS_ALLOCATED(p, log_name,                                    \
							  Kamanri::Utils::Result<T>(                      \
								  Kamanri::Utils::Result$::Status::EXCEPTION, \
								  code,                                       \
								  "The memory is not initialized"))
