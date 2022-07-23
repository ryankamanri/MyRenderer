#include <fstream>
#include "../../utils/logs.h"
#include "../../renderer/obj_reader.h"

using namespace Kamanri::Utils::Memory;
using namespace Kamanri::Renderer::ObjReader;
using namespace Kamanri::Utils::Result;
using namespace Kamanri::Utils::Logs;

constexpr const char *LOG_NAME = "Kamanri::Renderer::ObjReader";

DefaultResult ObjModel::Read(std::string const& file_name)
{
    std::ifstream fs(file_name);
    std::string str;

    if(file_name.compare(file_name.length() - 4, 4, ".obj") != 0)
    {
        auto message = "The file %s is not the type of .obj";
        Log::Error(LOG_NAME, message, file_name.c_str());
        return DEFAULT_RESULT_EXCEPTION(OBJ_READER_CODE_INVALID_TYPE, message);
    }

    if(!fs.good())
    {
        Log::Error(LOG_NAME, "Cannot Open The File %s", file_name.c_str());
        return DEFAULT_RESULT_EXCEPTION(OBJ_READER_CODE_CANNOT_READ_FILE, "Cannot Open The File");
    }

    while (std::getline(fs, str))
    {
        
        if(str.compare(0, 2, "vt") == 0)
        {
            Log::Trace(LOG_NAME, "met vt");
            continue;
        }
        if(str.compare(0, 2, "vn") == 0)
        {
            Log::Trace(LOG_NAME, "met vn");
            continue;
        }
        if(str.compare(0, 1, "v") == 0)
        {
            Log::Trace(LOG_NAME, "met v");
            continue;
        }
        if(str.compare(0, 1, "f") == 0)
        {
            Log::Trace(LOG_NAME, "met f");
            continue;
        }
    }

    fs.close();

    return DEFAULT_RESULT;
}