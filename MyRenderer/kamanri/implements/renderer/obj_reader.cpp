#include <fstream>
#include "../../utils/logs.h"
#include "../../renderer/obj_reader.h"
#include "../../utils/string.h"

using namespace Kamanri::Utils::Memory;
using namespace Kamanri::Renderer::ObjReader;
using namespace Kamanri::Utils::Result;
using namespace Kamanri::Utils::Logs;
using namespace Kamanri::Utils::String;

constexpr const char *LOG_NAME = "Kamanri::Renderer::ObjReader";

DefaultResult ObjModel::Read(std::string const &file_name)
{
    std::ifstream fs(file_name);
    std::string str;

    if (file_name.compare(file_name.length() - 4, 4, ".obj") != 0)
    {
        auto message = "The file %s is not the type of .obj";
        Log::Error(LOG_NAME, message, file_name.c_str());
        return DEFAULT_RESULT_EXCEPTION(OBJ_READER_CODE_INVALID_TYPE, message);
    }

    if (!fs.good())
    {
        Log::Error(LOG_NAME, "Cannot Open The File %s", file_name.c_str());
        return DEFAULT_RESULT_EXCEPTION(OBJ_READER_CODE_CANNOT_READ_FILE, "Cannot Open The File");
    }

    while (std::getline(fs, str))
    {
        auto splited_str_vec = Split(str, " ", true);
        auto vec_size = splited_str_vec.size();

        if (str.compare(0, 2, "vt") == 0)
        {
            try
            {
                auto vertice_texture = std::vector<double>();
                for (int i = 1; i < vec_size; i++)
                {
                    vertice_texture.push_back(std::stod(splited_str_vec[i]));
                }
                _vertice_textures.push_back(vertice_texture);
            }
            catch (std::exception e)
            {
                Log::Error(LOG_NAME, e.what());
                Log::Warn(LOG_NAME, "While str = '%s' and split str vec size = %d", str.c_str(), vec_size);
                return DEFAULT_RESULT_EXCEPTION(OBJ_READER_CODE_READING_EXCEPTION, e.what());
            }
            continue;
        }
        if (str.compare(0, 2, "vn") == 0)
        {
            try
            {
                _vertice_normals.push_back({std::stod(splited_str_vec[1]),
                                            std::stod(splited_str_vec[2]),
                                            std::stod(splited_str_vec[3])});
            }
            catch (std::exception e)
            {
                Log::Error(LOG_NAME, e.what());
                Log::Warn(LOG_NAME, "While str = '%s' and split str vec size = %d", str.c_str(), vec_size);
                return DEFAULT_RESULT_EXCEPTION(OBJ_READER_CODE_READING_EXCEPTION, e.what());
            }
            continue;
        }
        if (str.compare(0, 1, "v") == 0)
        {
            try
            {
                auto vertice = std::vector<double>();
                for (int i = 1; i < vec_size; i++)
                {
                    vertice.push_back(std::stod(splited_str_vec[i]));
                }
                _vertices.push_back(vertice);
            }
            catch (std::exception e)
            {
                Log::Error(LOG_NAME, e.what());
                Log::Warn(LOG_NAME, "While str = '%s' and split str vec size = %d", str.c_str(), vec_size);
                return DEFAULT_RESULT_EXCEPTION(OBJ_READER_CODE_READING_EXCEPTION, e.what());
            }

            continue;
        }
        if (str.compare(0, 1, "f") == 0)
        {
            auto face = Face();
            try
            {
                for (int i = 1; i < vec_size; i++)
                {
                    auto about_vertice_indexes = Split(splited_str_vec[i], "/");
                    auto indexes_size = about_vertice_indexes.size();

                    if (about_vertice_indexes[0].length() != 0)
                    {
                        face.vertice_indexes.push_back(std::stoi(about_vertice_indexes[0]));
                    }
                    if (indexes_size > 1 && about_vertice_indexes[1].length() != 0)
                    {
                        face.vertice_texture_indexes.push_back(std::stoi(about_vertice_indexes[1]));
                    }
                    if (indexes_size > 2 && about_vertice_indexes[2].length() != 0)
                    {
                        face.vertice_normal_indexes.push_back(std::stoi(about_vertice_indexes[2]));
                    }
                }
            }
            catch (std::exception e)
            {
                Log::Error(LOG_NAME, e.what());
                Log::Warn(LOG_NAME, "While str = '%s' and split str vec size = %d", str.c_str(), vec_size);
                return DEFAULT_RESULT_EXCEPTION(OBJ_READER_CODE_READING_EXCEPTION, e.what());
            }
            _faces.push_back(face);

            continue;
        }
    }

    fs.close();

    return DEFAULT_RESULT;
}

PMyResult<std::vector<double>> ObjModel::GetVertice(int index) const
{
    auto size = _vertices.size();
    if(size <= index)
    {
        Log::Error(LOG_NAME, "Index %d out of bound %d", index, size - 1);
        return RESULT_EXCEPTION(std::vector<double>, OBJ_READER_CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
    }
    return New<MyResult<std::vector<double>>>(_vertices[index]);
}

PMyResult<std::vector<double>> ObjModel::GetVerticeNormal(int index) const
{
    auto size = _vertice_normals.size();
    if(size <= index)
    {
        Log::Error(LOG_NAME, "Index %d out of bound %d", index, size - 1);
        return RESULT_EXCEPTION(std::vector<double>, OBJ_READER_CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
    }
    return New<MyResult<std::vector<double>>>(_vertice_normals[index]);
}

PMyResult<std::vector<double>> ObjModel::GetVerticeTexture(int index) const
{
    auto size = _vertice_textures.size();
    if(size <= index)
    {
        Log::Error(LOG_NAME, "Index %d out of bound %d", index, size - 1);
        return RESULT_EXCEPTION(std::vector<double>, OBJ_READER_CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
    }
    return New<MyResult<std::vector<double>>>(_vertice_textures[index]);
}

PMyResult<Face> ObjModel::GetFace(int index) const
{
    auto size = _faces.size();
    if(size <= index)
    {
        Log::Error(LOG_NAME, "Index %d out of bound %d", index, size - 1);
        return RESULT_EXCEPTION(Face, OBJ_READER_CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
    }
    return New<MyResult<Face>>(_faces[index]);
}