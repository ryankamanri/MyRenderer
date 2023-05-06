#pragma once
#ifndef SWIG
#include <vector>
#include "kamanri/maths/vector.hpp"
#include "kamanri/maths/smatrix.hpp"
#include "kamanri/renderer/tga_image.hpp"
#include "kamanri/utils/result_declare.hpp"
#endif
namespace Kamanri
{
	
	namespace Renderer
	{
		namespace World
		{
			namespace __
			{
				class Triangle3D;
			} // namespace __
			

			/**
			 * @brief The `Object` class is used to provide a handle of controlling the 3D object in class `World3D`.
			 * 
			 */
			class Object
			{
				private:
					std::vector<Kamanri::Maths::Vector>* _pvertices = nullptr;
					size_t _v_offset;
					size_t _v_length;
					size_t _t_offset;
					size_t _t_length;

					Kamanri::Renderer::TGAImage _img;
				public:
					// Object() = default;
					Object(std::vector<Kamanri::Maths::Vector>& vertices, size_t v_offset, size_t v_length, size_t t_offset, size_t t_length, std::string tga_image_name, bool is_use_cuda = false);
					void __UpdateTriangleRef(std::vector<Kamanri::Renderer::World::__::Triangle3D>& triangles, std::vector<Object>& objects, size_t index);
#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
					inline Kamanri::Renderer::TGAImage& GetImage() { return _img; }
					Kamanri::Utils::DefaultResult Transform(Kamanri::Maths::SMatrix const& transform_matrix) const;

					inline void DeleteCUDA() { _img.DeleteCUDA(); }
			};
			
			
		} // namespace World
		
	} // namespace Renderer
	
} // namespace Kamanri
