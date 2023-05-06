#pragma once
#include <stdio.h>
#include <stdarg.h>
#include <Windows.h>
#include <string>
#include "log_declare.hpp"

namespace Kamanri
{
	namespace Utils
	{

		template <typename... Ts>
#ifdef __CUDA_RUNTIME_H__  
		__host__ __device__
#endif
		inline int Print(const char* formatStr, Ts... argv)
		{
			return printf(formatStr, argv...);
		}


		template <typename... Ts>
#ifdef __CUDA_RUNTIME_H__  
		__host__ __device__
#endif
		inline int PrintLn(const char* formatStr, Ts... argv)
		{
			int retCode = printf(formatStr, argv...);
			printf("\n");
			return retCode;
		}

#ifdef __CUDA_RUNTIME_H__  
		__host__ __device__
#endif
		inline int PrintLn()
		{
			return printf("\n");
		}

		namespace __Log
		{
			constexpr const char* TRACE_SIGN = "trace";
			constexpr const char* DEBUG_SIGN = "debug";
			constexpr const char* INFO_SIGN = "infom";
			constexpr const char* WARN_SIGN = "warng";
			constexpr const char* ERROR_SIGN = "error";

			constexpr WORD DEFAULT_COLOR = 0x07;
			constexpr WORD TRACE_COLOR = 0x8F;
			constexpr WORD DEBUG_COLOR = 0x1F;
			constexpr WORD INFO_COLOR = 0x2F;
			constexpr WORD WARN_COLOR = 0x60;
			constexpr WORD ERROR_COLOR = 0x4F;

			template <typename... Ts>
			void Logger(WORD color, std::string sign, std::string name, std::string message, Ts... argv)
			{

				HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
				SetConsoleTextAttribute(handle, color);


				printf(sign.c_str());

				if (color == __Log::TRACE_COLOR || color == __Log::DEBUG_COLOR || color == __Log::INFO_COLOR)
				{
					SetConsoleTextAttribute(handle, __Log::DEFAULT_COLOR);
				}
				else
				{
					SetConsoleTextAttribute(handle, color >> 4);
				}

				printf(" [%s]: ", name.c_str());
				PrintLn(message.c_str(), argv...);

				SetConsoleTextAttribute(handle, __Log::DEFAULT_COLOR);

			}
		}

#define PRINT_LOCATION Kamanri::Utils::PrintLn("    at file %s, line %d.", __FILE__, __LINE__);


		/**
		 * @brief Log Class
		 * , Note that the `std::string` property should be converted to `char*`(use c_str() method)
		 *
		 */
		class Log
		{
			public:
			static LogLevel Level();
			static void SetLevel(LogLevel level);
			template <typename... Ts>
			static void Trace(std::string name, std::string message, Ts... argv)
			{
				if (Level() <= Log$::TRACE_LEVEL)
				{
					__Log::Logger(__Log::TRACE_COLOR, __Log::TRACE_SIGN, name, message, argv...);
				}
			}
			template <typename... Ts>
			static void Debug(std::string name, std::string message, Ts... argv)
			{
				if (Level() <= Log$::DEBUG_LEVEL)
				{
					__Log::Logger(__Log::DEBUG_COLOR, __Log::DEBUG_SIGN, name, message, argv...);
				}
			}
			template <typename... Ts>
			static void Info(std::string name, std::string message, Ts... argv)
			{
				if (Level() <= Log$::INFO_LEVEL)
				{
					__Log::Logger(__Log::INFO_COLOR, __Log::INFO_SIGN, name, message, argv...);
				}
			}
			template <typename... Ts>
			static void Warn(std::string name, std::string message, Ts... argv)
			{
				if (Level() <= Log$::WARN_LEVEL)
				{
					__Log::Logger(__Log::WARN_COLOR, __Log::WARN_SIGN, name, message, argv...);
				}
			}
			template <typename... Ts>
			static void Error(std::string name, std::string message, Ts... argv)
			{
				if (Level() <= Log$::ERROR_LEVEL)
				{
					__Log::Logger(__Log::ERROR_COLOR, __Log::ERROR_SIGN, name, message, argv...);
				}
			}

			template <LogLevel LOG_LEVEL, typename... Ts>
			static void Print(const char* formatStr, Ts... argv)
			{
				if (Level() > LOG_LEVEL) return;
				printf(formatStr, argv...);
			}

			template <LogLevel LOG_LEVEL, typename... Ts>
			static void PrintLn(const char* formatStr, Ts... argv)
			{
				if (Level() > LOG_LEVEL) return;
				printf(formatStr, argv...);
				printf("\n");
			}
		};


	}
}

