#include "kamanri/utils/logs.hpp"

using namespace Kamanri::Utils;

LogLevel Log::_level;

LogLevel Log::Level()
{
    return _level;
}

void Log::SetLevel(LogLevel level) {
    _level = level;
}

