#include "../../utils/logs.h"

using namespace Kamanri::Utils::Logs;

LogLevel Log::_level;

void Log::Level(LogLevel level) {
    _level = level;
}

