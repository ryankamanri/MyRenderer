#include <fstream>
#include "kamanri/utils/log.hpp"
#include "kamanri/renderer/obj_model.hpp"
#include "kamanri/utils/string.hpp"
#include "kamanri/utils/result.hpp"

using namespace Kamanri::Utils;
using namespace Kamanri::Renderer;
using namespace Kamanri::Utils::String;

namespace Kamanri
{
	namespace Renderer
	{

		namespace __ObjModel
		{
			constexpr const char *LOG_NAME = STR(Kamanri::Renderer::ObjModel);
		} // namespace ObjModel$

	} // namespace Renderer

} // namespace Kamanri


ObjModel::ObjModel(std::string const& file_name, std::string const& tga_file_name)
{
	auto result = ReadObjFileAndInit(file_name);
	if(result.IsException())
	{
		Log::Error(__ObjModel::LOG_NAME, "An Exception occured while initing the ObjModel:");
		result.Print();
		exit(result.Code());
	}
	if(tga_file_name.empty()) return;
	_tga_image_name = tga_file_name;
	// if(!_img.ReadTGAFile(tga_file_name))
	// {
	//     Log::Error(__ObjModel::LOG_NAME, "Cannot read the TGA file '%s'.", tga_file_name.c_str());
	// }
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

Result<std::vector<double>> ObjModel::GetVertex(size_t index) const
{
	auto size = _vertices.size();
	if (size <= index)
	{
		Log::Error(__ObjModel::LOG_NAME, "Index %d out of bound %d", index, size - 1);
		PRINT_LOCATION;
		return RESULT_EXCEPTION(std::vector<double>, ObjModel$::CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
	}
	return Result<std::vector<double>>(_vertices[index]);
}

Result<std::vector<double>> ObjModel::GetVertexNormal(size_t index) const
{
	auto size = _vertex_normals.size();
	if (size <= index)
	{
		Log::Error(__ObjModel::LOG_NAME, "Index %d out of bound %d", index, size - 1);
		PRINT_LOCATION;
		return RESULT_EXCEPTION(std::vector<double>, ObjModel$::CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
	}
	return Result<std::vector<double>>(_vertex_normals[index]);
}

Result<std::vector<double>> ObjModel::GetVertexTexture(size_t index) const
{
	auto size = _vertex_textures.size();
	if (size <= index)
	{
		Log::Error(__ObjModel::LOG_NAME, "Index %d out of bound %d", index, size - 1);
		PRINT_LOCATION;
		return RESULT_EXCEPTION(std::vector<double>, ObjModel$::CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
	}
	return Result<std::vector<double>>(_vertex_textures[index]);
}

Result<ObjModel$::Face> ObjModel::GetFace(size_t index) const
{
	auto size = _faces.size();
	if (size <= index)
	{
		Log::Error(__ObjModel::LOG_NAME, "Index %d out of bound %d", index, size - 1);
		PRINT_LOCATION;
		return RESULT_EXCEPTION(ObjModel$::Face, ObjModel$::CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
	}
	return Result<ObjModel$::Face>(_faces[index]);
}




DefaultResult ObjModel::ReadObjFileAndInit(std::string const &file_name)
{
	std::ifstream fs(file_name);
	std::string str;

	if (file_name.compare(file_name.length() - 4, 4, ".obj") != 0)
	{
		auto message = "The file %s is not the type of .obj";
		Log::Error(__ObjModel::LOG_NAME, message, file_name.c_str());
		PRINT_LOCATION;
		return DEFAULT_RESULT_EXCEPTION(ObjModel$::CODE_INVALID_TYPE, message);
	}

	if (!fs.good())
	{
		Log::Error(__ObjModel::LOG_NAME, "Cannot Open The File %s", file_name.c_str());
		PRINT_LOCATION;
		return DEFAULT_RESULT_EXCEPTION(ObjModel$::CODE_CANNOT_READ_FILE, "Cannot Open The File");
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
				for (size_t i = 1; i < vec_size; i++)
				{
					vertex_texture.push_back(std::stod(splited_str_vec[i]));
				}
				_vertex_textures.push_back(vertex_texture);
			}
			catch (std::exception e)
			{
				Log::Error(__ObjModel::LOG_NAME, e.what());
				Log::Warn(__ObjModel::LOG_NAME, "While str = '%s' and split str vec size = %d", str.c_str(), vec_size);
				PRINT_LOCATION;
				return DEFAULT_RESULT_EXCEPTION(ObjModel$::CODE_READING_EXCEPTION, e.what());
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
				Log::Error(__ObjModel::LOG_NAME, e.what());
				Log::Warn(__ObjModel::LOG_NAME, "While str = '%s' and split str vec size = %d", str.c_str(), vec_size);
				PRINT_LOCATION;
				return DEFAULT_RESULT_EXCEPTION(ObjModel$::CODE_READING_EXCEPTION, e.what());
			}
			continue;
		}
		if (str.compare(0, 1, "v") == 0)
		{
			try
			{
				auto vertex = std::vector<double>();
				for (size_t i = 1; i < vec_size; i++)
				{
					vertex.push_back(std::stod(splited_str_vec[i]));
				}
				_vertices.push_back(vertex);
			}
			catch (std::exception e)
			{
				Log::Error(__ObjModel::LOG_NAME, e.what());
				Log::Warn(__ObjModel::LOG_NAME, "While str = '%s' and split str vec size = %d", str.c_str(), vec_size);
				PRINT_LOCATION;
				return DEFAULT_RESULT_EXCEPTION(ObjModel$::CODE_READING_EXCEPTION, e.what());
			}

			continue;
		}
		if (str.compare(0, 1, "f") == 0)
		{
			auto face = ObjModel$::Face();
			try
			{
				for (size_t i = 1; i < vec_size; i++)
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
				Log::Error(__ObjModel::LOG_NAME, e.what());
				Log::Warn(__ObjModel::LOG_NAME, "While str = '%s' and split str vec size = %d", str.c_str(), vec_size);
				PRINT_LOCATION;
				return DEFAULT_RESULT_EXCEPTION(ObjModel$::CODE_READING_EXCEPTION, e.what());
			}
			_faces.push_back(face);

			continue;
		}
	}

	fs.close();

	return DEFAULT_RESULT;
}