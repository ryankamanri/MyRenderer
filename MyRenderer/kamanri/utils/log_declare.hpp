#pragma once

namespace Kamanri
{
	namespace Utils
	{
		using LogLevel = int;

		namespace Log$
		{
			constexpr LogLevel TRACE_LEVEL = 0;
			constexpr LogLevel DEBUG_LEVEL = 1;
			constexpr LogLevel INFO_LEVEL = 2;
			constexpr LogLevel WARN_LEVEL = 3;
			constexpr LogLevel ERROR_LEVEL = 4;

		}

		class Log;
	} // namespace Utils
	
} // namespace Kamanri
