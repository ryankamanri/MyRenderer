#pragma once
#include <string>
#include <vector>

namespace Kamanri
{
    namespace Utils
    {
        namespace String
        {
            std::vector<std::string> Split(std::string const& str, std::string const& delim, bool is_remove_empty = false);
        }
    }
}