#include "../../utils/logs.hpp"

using namespace Kamanri::Utils::Logs;

LogLevel Log::_level;

void Log::Level(LogLevel level) {
    _level = level;
}

