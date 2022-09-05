#include <fstream>
#include "../../utils/logs.hpp"
#include "../../renderer/obj_reader.hpp"
#include "../../utils/string.hpp"

using namespace Kamanri::Utils::Memory;
using namespace Kamanri::Renderer::ObjReader;
using namespace Kamanri::Utils::Result;
using namespace Kamanri::Utils::Logs;
using namespace Kamanri::Utils::String;

constexpr const char *LOG_NAME = STR(Kamanri::Renderer::ObjReader);

ObjModel::ObjModel(std::string const& file_name)
{
    auto result = Read(file_name);
    if(result->IsException())
    {
        Log::Error(LOG_NAME, "An Exception occured while initing the ObjModel:");
        result->Print();
    }
}

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
                auto vertex_texture = std::vector<double>();
                for (int i = 1; i < vec_size; i++)
                {
                    vertex_texture.push_back(std::stod(splited_str_vec[i]));
                }
                _vertex_textures.push_back(vertex_texture);
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
                _vertex_normals.push_back({std::stod(splited_str_vec[1]),
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
                auto vertex = std::vector<double>();
                for (int i = 1; i < vec_size; i++)
                {
                    vertex.push_back(std::stod(splited_str_vec[i]));
                }
                _vertices.push_back(vertex);
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
                    auto about_vertex_indexes = Split(splited_str_vec[i], "/");
                    auto indexes_size = about_vertex_indexes.size();

                    if (about_vertex_indexes[0].length() != 0)
                    {
                        face.vertex_indexes.push_back(std::stoi(about_vertex_indexes[0]));
                    }
                    if (indexes_size > 1 && about_vertex_indexes[1].length() != 0)
                    {
                        face.vertex_texture_indexes.push_back(std::stoi(about_vertex_indexes[1]));
                    }
                    if (indexes_size > 2 && about_vertex_indexes[2].length() != 0)
                    {
                        face.vertex_normal_indexes.push_back(std::stoi(about_vertex_indexes[2]));
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

size_t ObjModel::GetVertexSize() const
{
    return _vertices.size();
}
size_t ObjModel::GetVertexNormalSize() const
{
    return _vertex_normals.size();
}
size_t ObjModel::GetVertexTextureSize() const
{
    return _vertex_textures.size();
}
size_t ObjModel::GetFaceSize() const
{
    return _faces.size();
}

PMyResult<std::vector<double>> ObjModel::GetVertex(int index) const
{
    auto size = _vertices.size();
    if (size <= index)
    {
        Log::Error(LOG_NAME, "Index %d out of bound %d", index, size - 1);
        return RESULT_EXCEPTION(std::vector<double>, OBJ_READER_CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
    }
    return New<MyResult<std::vector<double>>>(_vertices[index]);
}

PMyResult<std::vector<double>> ObjModel::GetVertexNormal(int index) const
{
    auto size = _vertex_normals.size();
    if (size <= index)
    {
        Log::Error(LOG_NAME, "Index %d out of bound %d", index, size - 1);
        return RESULT_EXCEPTION(std::vector<double>, OBJ_READER_CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
    }
    return New<MyResult<std::vector<double>>>(_vertex_normals[index]);
}

PMyResult<std::vector<double>> ObjModel::GetVertexTexture(int index) const
{
    auto size = _vertex_textures.size();
    if (size <= index)
    {
        Log::Error(LOG_NAME, "Index %d out of bound %d", index, size - 1);
        return RESULT_EXCEPTION(std::vector<double>, OBJ_READER_CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
    }
    return New<MyResult<std::vector<double>>>(_vertex_textures[index]);
}

PMyResult<Face> ObjModel::GetFace(int index) const
{
    auto size = _faces.size();
    if (size <= index)
    {
        Log::Error(LOG_NAME, "Index %d out of bound %d", index, size - 1);
        return RESULT_EXCEPTION(Face, OBJ_READER_CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
    }
    return New<MyResult<Face>>(_faces[index]);
}