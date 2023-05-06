#include "kamanri/utils/log.hpp"

using namespace Kamanri::Utils;
namespace Kamanri
{
	namespace Utils
	{
		namespace __Log
		{
			LogLevel _level = Log$::DEBUG_LEVEL;
		} // namespace __Log
	} // namespace Utils

} // namespace Kamanri



LogLevel Log::Level()
{
	return __Log::_level;
}

void Log::SetLevel(LogLevel level)
{
	__Log::_level = level;
}

