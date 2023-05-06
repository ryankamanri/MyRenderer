#pragma once
#ifndef SWIG
#include "kamanri/renderer/world/world3d$.hpp"
#endif

namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{	

			class World3D
			{
			private:
				/* data */
				Kamanri::Renderer::World::Camera _camera;
				/// Hint Whether CUDA accelerated
				Kamanri::Renderer::World::__::Configs _configs;
				/// @brief Store all resources
				Kamanri::Renderer::World::__::Resources _resources;
				/// @brief Store all envirment objects
				Kamanri::Renderer::World::__::Environment _environment;
				/// @brief Store all buffers
				Kamanri::Renderer::World::__::Buffers _buffers;

				World3D* _cuda_world;

			public:
				World3D(Kamanri::Renderer::World::Camera&& camera, Kamanri::Renderer::World::BlinnPhongReflectionModel&& model, bool is_use_cuda = false);
				~World3D();
				World3D& operator=(World3D const& other);
				World3D& operator=(World3D&& other);
				Kamanri::Renderer::World::Camera& GetCamera() { return _camera; }
				Kamanri::Utils::Result<Object *> AddObjModel(Kamanri::Renderer::ObjModel const &model);
				World3D& AddObjModel(Kamanri::Renderer::ObjModel const &model, Kamanri::Maths::SMatrix const& transform_matrix);
				World3D& Commit();
				Kamanri::Utils::DefaultResult Build();
#ifdef __CUDA_RUNTIME_H__  
				__device__
#endif
				void __BuildForPixel(size_t x, size_t y);
				Kamanri::Renderer::World::FrameBuffer const& GetFrameBuffer(int x, int y);
				inline DWORD* Bitmap() { return _buffers.GetBitmapBufferPtr(); }
			};
			
			
		} // namespace World3Ds
		
	} // namespace Renderer
	
} // namespace Kamanri
