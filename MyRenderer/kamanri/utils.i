%module Utils
%feature("python:annotations", "c");

%{
#include "kamanri/utils/log.hpp"
%}

namespace Kamanri
{
	namespace Utils
	{
		using LogLevel = int;
		class Log
		{
			public:
			static Kamanri::Utils::LogLevel Level();
			static void SetLevel(Kamanri::Utils::LogLevel level);
		};
	}
}
