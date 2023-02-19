#include "kamanri/utils/result.hpp"
#include <fstream>

using namespace Kamanri::Utils::Result$;

StackTrace::StackTrace(std::string const& file, int line, int lineCount) : _file(file), _line(line), _line_count(lineCount)
{
	// auto get the statement

	// read the source file
	std::ifstream fs(file);

	std::string str;
	std::string statement;
	int index = 1;
	while (index < line && getline(fs, str))
	{
		index++;
	}
	while (index < line + lineCount && getline(fs, str))
	{
		statement += "\n\t\t";
		statement += str;
		index++;
	}

	fs.close();
	if (index < line)
	{
		Log::Error("MyResultStackTraceException", "File %s Lines Count %d < %d (The Given)", file.c_str(), index, line);
		Log::Warn("MyResultStackTraceException",
				  "It Might Due To Your Code Source File '%s' Has Been Occupied By Another Process, Try To Close All Process Which Keep Occupying It, And The Following StackTrace Might Be In A Mess, Do NOT Care.",
				  file.c_str());
				  PRINT_LOCATION;
		return;
	}

	_statement = statement;
}

void StackTrace::Print() const
{
	PrintLn("\tat %s, Line %d - %d", this->_file.c_str(), this->_line, this->_line + this->_line_count);
	if (!_statement.empty())
	{
		PrintLn("\t\t%s", _statement.c_str());
	}
}