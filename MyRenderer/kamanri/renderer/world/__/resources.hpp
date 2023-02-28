#pragma once

namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{
			namespace __
			{
				class Resources
				{
					private:
					/* data */
					public:
					Resources() = default;
					/**
				 * @brief Used to store every vertex, note that the cluster of vertices of a object is stored in order.
				 *
				 */
					std::vector<Maths::Vector> vertices;
					/**
					 * @brief Used to store every PROJECTION transformed vertex, note that the cluster of vertices of a object is stored in order.
					 *
					 */
					std::vector<Maths::Vector> vertices_transformed;
					/// @brief Used to store ONLY MODEL VIEW transformed vertices
					std::vector<Maths::Vector> vertices_model_view_transformed;
					/// @brief 顶点纹理，代表当前顶点对应纹理图的哪个像素，通常是0-1，如果大于1，就相当于将纹理重新扩充然后取值，比如镜像填充、翻转填充之类的，然后根据纹理图的宽高去计算具体像素位置
					std::vector<Maths::Vector> vertex_textures;
					/// @brief 顶点法线，物理里面有说过眼睛看到物体是因为光线经过物体表面反射到眼睛，所以这个法线就是通过入射光线计算反射光线使用的法线。
					std::vector<Maths::Vector> vertex_normals;
					/// @brief Used to store ONLY MODEL VIEW transformed vertex normals
					std::vector<Maths::Vector> vertex_normals_model_view_transformed;

					Resources& operator=(Resources&& other)
					{
						vertices = std::move(other.vertices);
						vertices_transformed = std::move(other.vertices_transformed);
						vertices_model_view_transformed = std::move(other.vertices_model_view_transformed);
						vertex_textures = std::move(other.vertex_textures);
						vertex_normals = std::move(other.vertex_normals);
						vertex_normals_model_view_transformed = std::move(other.vertex_normals_model_view_transformed);
						return *this;
					}
				};
			} // namespace __

		} // namespace World

	} // namespace Renderer

} // namespace Kamanri
