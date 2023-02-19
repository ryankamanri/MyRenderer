#include "cuda_dll/src/build_world.cuh"
#include "kamanri/utils/logs.hpp"
#include "kamanri/renderer/world/__/buffers.hpp"
#include "cuda_dll/src/utils/cuda_thread_config.cuh"

using namespace Kamanri::Utils;
using namespace Kamanri::Renderer::World;

namespace BuildWorld$
{
	constexpr const char* LOG_NAME = STR(BuildWorld);

	constexpr const int CODE_NORM = 0;
} // namespace BuildWorld$

BuildWorldCode BuildWorld(World3D* p_world)
{
	using namespace BuildWorld$;
	Log::Debug(LOG_NAME, "Called of BuildWorld!");
	p_world->Build();
	return CODE_NORM;

}

////////////////////////////////////////////////////////////////////////////
namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{
			namespace __
			{

			} // namespace __

			namespace __World3D
			{
				constexpr const char* LOG_NAME = STR(BuildWorld$) STR(Kamanri::Renderer::World::World3D);


				BuildWorldCode BuildTriangles(std::vector<__::Triangle3D>& triangles, __::Resources& resources)
				{
					Log::Debug(BuildWorld$::LOG_NAME, "triangles at %p", &triangles);
					return BuildWorld$::CODE_NORM;
				}

				BuildWorldCode WriteTo(std::vector<__::Triangle3D>& triangles, __::Buffers& buffers, double nearest_dist)
				{
					return BuildWorld$::CODE_NORM;
				}
			} // namespace __World3D


			DefaultResult World3D::Build()
			{
				using namespace __World3D;

				// WriteTo(triangles, buffers, nearest_dist);
				return DEFAULT_RESULT;
			}
		} // namespace World

	} // namespace Renderer

} // namespace Kamanri







