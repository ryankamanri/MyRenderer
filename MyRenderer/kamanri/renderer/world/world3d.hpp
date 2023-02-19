#pragma once
#include <vector>
#include "camera.hpp"
#include "kamanri/renderer/obj_model.hpp"
#include "object.hpp"
#include "__/triangle3d.hpp"
#include "__/environment.hpp"
#include "__/buffers.hpp"
#include "__/resources.hpp"
#include "kamanri/maths/vector.hpp"
#include "kamanri/maths/matrix.hpp"

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
				
				/// @brief Store all resources
				__::Resources _resources;
				/// @brief Store all envirment objects
				__::Environment _environment;
				/// @brief Store all buffers
				__::Buffers _buffers;

			public:
				World3D();
				World3D(Camera&& camera);
				World3D& operator=(World3D&& other);
				Camera& GetCamera() { return _camera; }
				Utils::Result<Object *> AddObjModel(ObjModel const &model);
				World3D&& AddObjModel(ObjModel const &model, Maths::SMatrix const& transform_matrix);
				Utils::DefaultResult Build();
				FrameBuffer const& FrameBuffer(int x, int y);
				inline DWORD* Bitmap() { return _buffers.GetBitmapBufferPtr(); }
			};
			
			
		} // namespace World3Ds
		
	} // namespace Renderer
	
} // namespace Kamanri
