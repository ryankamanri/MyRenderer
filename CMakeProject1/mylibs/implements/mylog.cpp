#include "../mylog.h"

LogLevel Log::_level;

void Log::Level(LogLevel level) {
    _level = level;
}

