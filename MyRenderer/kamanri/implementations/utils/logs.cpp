#include "kamanri/utils/logs.hpp"

using namespace Kamanri::Utils;
namespace __Logs
{
	LogLevel _level;
} // namespace __Logs

LogLevel Log::Level()
{
	return __Logs::_level;
}

void Log::SetLevel(LogLevel level) {
	__Logs::_level = level;
}

