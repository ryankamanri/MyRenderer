#pragma once
#include <stdio.h>
#include <stdarg.h>
#include <Windows.h>
#include <string>

namespace Kamanri
{
    namespace Utils
    {
        template <typename... Ts>
        inline int Print(const char *formatStr, Ts... argv)
        {
            return printf(formatStr, argv...);
        }

        template <typename... Ts>
        inline int PrintLn(const char *formatStr, Ts... argv)
        {
            int retCode = printf(formatStr, argv...);
            printf("\n");
            return retCode;
        }

        inline int PrintLn()
        {
            return printf("\n");
        }

        using LogLevel = int;

        namespace Log$
        {
            constexpr int TRACE_LEVEL = 0;
            constexpr int DEBUG_LEVEL = 1;
            constexpr int INFO_LEVEL = 2;
            constexpr int WARN_LEVEL = 3;
            constexpr int ERROR_LEVEL = 4;

        }

        namespace __Log
        {
            constexpr const char *TRACE_SIGN = "trace";
            constexpr const char *DEBUG_SIGN = "debug";
            constexpr const char *INFO_SIGN = "infor";
            constexpr const char *WARN_SIGN = "warng";
            constexpr const char *ERROR_SIGN = "error";

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

        /**
         * @brief Log Class
         *
         */
        class Log
        {
        private:
            static LogLevel _level;

        public:
            static void Level(LogLevel level);
            template <typename... Ts>
            static void Trace(std::string name, std::string message, Ts... argv);
            template <typename... Ts>
            static void Debug(std::string name, std::string message, Ts... argv);
            template <typename... Ts>
            static void Info(std::string name, std::string message, Ts... argv);
            template <typename... Ts>
            static void Warn(std::string name, std::string message, Ts... argv);
            template <typename... Ts>
            static void Error(std::string name, std::string message, Ts... argv);
        };

        template <typename... Ts>
        void Log::Trace(std::string name, std::string message, Ts... argv)
        {
            if (_level <= Log$::TRACE_LEVEL)
            {
                __Log::Logger(__Log::TRACE_COLOR, __Log::TRACE_SIGN, name, message, argv...);
            }
        }

        template <typename... Ts>
        void Log::Debug(std::string name, std::string message, Ts... argv)
        {
            if (_level <= Log$::DEBUG_LEVEL)
            {
                __Log::Logger(__Log::DEBUG_COLOR, __Log::DEBUG_SIGN, name, message, argv...);
            }
        }

        template <typename... Ts>
        void Log::Info(std::string name, std::string message, Ts... argv)
        {
            if (_level <= Log$::INFO_LEVEL)
            {
                __Log::Logger(__Log::INFO_COLOR, __Log::INFO_SIGN, name, message, argv...);
            }
        }

        template <typename... Ts>
        void Log::Warn(std::string name, std::string message, Ts... argv)
        {
            if (_level <= Log$::WARN_LEVEL)
            {
                __Log::Logger(__Log::WARN_COLOR, __Log::WARN_SIGN, name, message, argv...);
            }
        }

        template <typename... Ts>
        void Log::Error(std::string name, std::string message, Ts... argv)
        {
            if (_level <= Log$::ERROR_LEVEL)
            {
                __Log::Logger(__Log::ERROR_COLOR, __Log::ERROR_SIGN, name, message, argv...);
            }
        }

    }
}
