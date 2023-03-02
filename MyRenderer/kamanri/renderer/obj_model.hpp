#pragma once
#include <vector>
#include "kamanri/utils/result_declare.hpp"
#include "tga_image.hpp"

namespace Kamanri
{
	namespace Renderer
	{
		namespace ObjModel$
		{
			constexpr int CODE_INVALID_TYPE = 0;
			constexpr int CODE_CANNOT_READ_FILE = 100;
			constexpr int CODE_READING_EXCEPTION = 200;
			constexpr int CODE_INDEX_OUT_OF_BOUND = 300;

			class Face
			{
			public:
				std::vector<int> vertex_indexes;
				std::vector<int> vertex_texture_indexes;
				std::vector<int> vertex_normal_indexes;
			};
		}

		class ObjModel
		{
		public:
			explicit ObjModel(std::string const &file_name, std::string const& tga_file_name = "");
			size_t GetVertexSize() const;
			size_t GetVertexNormalSize() const;
			size_t GetVertexTextureSize() const;
			size_t GetFaceSize() const;
			Utils::Result<std::vector<double>> GetVertex(size_t index) const;
			Utils::Result<std::vector<double>> GetVertexNormal(size_t index) const;
			Utils::Result<std::vector<double>> GetVertexTexture(size_t index) const;
			Utils::Result<ObjModel$::Face> GetFace(size_t index) const;

			inline std::string GetTGAImageName() const { return _tga_image_name; }
			

		private:
			std::vector<std::vector<double>> _vertices;
			/// @brief 顶点法线，物理里面有说过眼睛看到物体是因为光线经过物体表面反射到眼睛，所以这个法线就是通过入射光线计算反射光线使用的法线。
			std::vector<std::vector<double>> _vertex_normals;
			/// @brief 顶点纹理，代表当前顶点对应纹理图的哪个像素，通常是0-1，如果大于1，就相当于将纹理重新扩充然后取值，比如镜像填充、翻转填充之类的，然后根据纹理图的宽高去计算具体像素位置
			std::vector<std::vector<double>> _vertex_textures;
			std::vector<ObjModel$::Face> _faces;

			std::string _tga_image_name;

			Utils::DefaultResult ReadObjFileAndInit(std::string const &file_name);
		};

	}
}