#pragma once
#include <stdio.h>
#include <stdarg.h>
#include <Windows.h>

template<typename... Ts>
int Print(const char* formatStr, Ts... argv) {
    return printf(formatStr, argv...);
}

template<typename... Ts>
int PrintLn(const char* formatStr, Ts... argv) {
    int retCode = printf(formatStr, argv...);
    printf("\n");
    return retCode;
}

typedef int LogLevel;

#define TRACE_LEVEL 0
#define DEBUG_LEVEL 1
#define INFO_LEVEL 2
#define WARN_LEVEL 3
#define ERROR_LEVEL 4


#define TRACE_SIGN "TRACE"
#define DEBUG_SIGN "DEBUG"
#define INFO_SIGN "INFOR"
#define WARN_SIGN "WARNG"
#define ERROR_SIGN "ERROR"

#define DEFAULT_COLOR 0x07
#define TRACE_COLOR 0x8F
#define DEBUG_COLOR 0x1F
#define INFO_COLOR 0x2F
#define WARN_COLOR 0x6F
#define ERROR_COLOR 0x4F

/**
 * @brief The Level Of The Logger, Use Log<>::Level To Change It.
 * 
 */
LogLevel _level = TRACE_LEVEL;

template<typename... Ts>
class Log
{
    public:
        static void Level(LogLevel level);
        static void Trace(const char* name, const char* message, Ts... argv);
        static void Debug(const char* name, const char* message, Ts... argv);
        static void Info(const char* name, const char* message, Ts... argv);
        static void Warn(const char* name, const char* message, Ts... argv);
        static void Error(const char* name, const char* message, Ts... argv);
};



template<typename... Ts>
void Log<Ts...>::Level(LogLevel level) {
    _level = level;
}


template<typename... Ts>
void Logger(int color, const char* sign, const char* name, const char* message, Ts... argv) {

    HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(handle, color);
    printf(sign);
    if(color == TRACE_COLOR || color == DEBUG_COLOR || color == INFO_COLOR) {
        SetConsoleTextAttribute(handle, DEFAULT_COLOR);
    } else {
        SetConsoleTextAttribute(handle, color >> 4);
    }
    
    printf(" [%s]: ", name);
    PrintLn(message, argv...);
    
    SetConsoleTextAttribute(handle, DEFAULT_COLOR);
}

template<typename... Ts>
void Log<Ts...>::Trace(const char* name, const char* message, Ts... argv) {
    if(_level <= TRACE_LEVEL) {
        Logger(TRACE_COLOR, TRACE_SIGN, name, message, argv...);
    }
}

template<typename... Ts>
void Log<Ts...>::Debug(const char* name, const char* message, Ts... argv) {
    if(_level <= DEBUG_LEVEL) {
        Logger(DEBUG_COLOR, DEBUG_SIGN, name, message, argv...);
    }
}

template<typename... Ts>
void Log<Ts...>::Info(const char* name, const char* message, Ts... argv) {
    if(_level <= INFO_LEVEL) {
        Logger(INFO_COLOR, INFO_SIGN, name, message, argv...);
    }
    
}

template<typename... Ts>
void Log<Ts...>::Warn(const char* name, const char* message, Ts... argv) {
    if(_level <= WARN_LEVEL) {
        Logger(WARN_COLOR, WARN_SIGN, name, message, argv...);
    }
}

template<typename... Ts>
void Log<Ts...>::Error(const char* name, const char* message, Ts... argv) {
    if(_level <= ERROR_LEVEL) {
        Logger(ERROR_COLOR, ERROR_SIGN, name, message, argv...);
    }
}



