#include "kamanri/utils/logs.hpp"
#include "cuda_dll/src/set_log_level.cuh"

using namespace Kamanri::Utils;

namespace __SetLogLevel
{
	LogLevel _level;
}


LogLevel Log::Level()
{
	return __SetLogLevel::_level;
}

void Log::SetLevel(LogLevel level) {
	__SetLogLevel::_level = level;
}

void SetLogLevel(LogLevel level)
{
	Log::SetLevel(level);
}
