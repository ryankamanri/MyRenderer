#include "../../utils/logs.hpp"

using namespace Kamanri::Utils;

LogLevel Log::_level;

void Log::Level(LogLevel level) {
    _level = level;
}

