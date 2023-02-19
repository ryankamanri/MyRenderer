#include "kamanri/utils/logs.hpp"
#include "cuda_dll/src/set_log_level.cuh"

using namespace Kamanri::Utils;

LogLevel Kamanri::Utils::Log::_level;

LogLevel Log::Level()
{
	return _level;
}

void Log::SetLevel(LogLevel level) {
	_level = level;
}

// TODO:

void SetLogLevel(LogLevel level)
{
	Log::SetLevel(level);
}
