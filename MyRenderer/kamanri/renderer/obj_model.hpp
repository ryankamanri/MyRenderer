#pragma once
#ifndef SWIG
#include "kamanri/renderer/obj_model$.hpp"
#endif

namespace Kamanri
{
	namespace Renderer
	{

		class ObjModel
		{
		public:
			explicit ObjModel(std::string const &file_name, std::string const& tga_file_name = "");
			size_t GetVertexSize() const;
			size_t GetVertexNormalSize() const;
			size_t GetVertexTextureSize() const;
			size_t GetFaceSize() const;
			Kamanri::Utils::Result<std::vector<double>> GetVertex(size_t index) const;
			Kamanri::Utils::Result<std::vector<double>> GetVertexNormal(size_t index) const;
			Kamanri::Utils::Result<std::vector<double>> GetVertexTexture(size_t index) const;
			Kamanri::Utils::Result<Kamanri::Renderer::ObjModel$::Face> GetFace(size_t index) const;

			inline std::string GetTGAImageName() const { return _tga_image_name; }
			

		private:
			std::vector<std::vector<double>> _vertices;
			/// @brief 顶点法线，物理里面有说过眼睛看到物体是因为光线经过物体表面反射到眼睛，所以这个法线就是通过入射光线计算反射光线使用的法线。
			std::vector<std::vector<double>> _vertex_normals;
			/// @brief 顶点纹理，代表当前顶点对应纹理图的哪个像素，通常是0-1，如果大于1，就相当于将纹理重新扩充然后取值，比如镜像填充、翻转填充之类的，然后根据纹理图的宽高去计算具体像素位置
			std::vector<std::vector<double>> _vertex_textures;
			std::vector<Kamanri::Renderer::ObjModel$::Face> _faces;

			std::string _tga_image_name;

			Kamanri::Utils::DefaultResult ReadObjFileAndInit(std::string const &file_name);
		};

	}
}