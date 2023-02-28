#pragma once
#include <vector>
#include "camera.hpp"
#include "kamanri/renderer/obj_model.hpp"
#include "object.hpp"
#include "__/all.hpp"
#include "kamanri/maths/all.hpp"

namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{

			namespace World3D$
			{
				constexpr int CODE_UNHANDLED_EXCEPTION = 0;
			} // namespace World3D$
			

			class World3D
			{
			private:
				/* data */
				Camera _camera;
				/// Hint Whether CUDA accelerated
				__::Configs _configs;
				/// @brief Store all resources
				__::Resources _resources;
				/// @brief Store all envirment objects
				__::Environment _environment;
				/// @brief Store all buffers
				__::Buffers _buffers;

				void BuildForPixel(size_t i, size_t j);

			public:
				World3D(Camera&& camera, BlingPhongReflectionModel&& model);
				~World3D();
				World3D& operator=(World3D&& other);
				Camera& GetCamera() { return _camera; }
				Utils::Result<Object *> AddObjModel(ObjModel const &model);
				World3D& AddObjModel(ObjModel const &model, Maths::SMatrix const& transform_matrix);
				World3D& Commit(bool is_use_cuda = false);
				Utils::DefaultResult Build();
				FrameBuffer const& FrameBuffer(int x, int y);
				inline DWORD* Bitmap() { return _buffers.GetBitmapBufferPtr(); }
			};
			
			
		} // namespace World3Ds
		
	} // namespace Renderer
	
} // namespace Kamanri
