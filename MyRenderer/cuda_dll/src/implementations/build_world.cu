#include <cuda_runtime.h>
#include "cuda_dll/src/build_world.cuh"
#include "cuda_dll/src/utils/cuda_thread_config.cuh"
#include "cuda_dll/src/utils/cuda_error_check.cuh"

// implementations in device code
#include "renderer/world/world3d.impl.cuh"


using namespace Kamanri::Utils;
using namespace Kamanri::Maths;
using namespace Kamanri::Renderer;
using namespace Kamanri::Renderer::World;
using namespace Kamanri::Renderer::World::__;

namespace __BuildWorld
{
	constexpr const char* LOG_NAME = STR(BuildWorld);

} // namespace BuildWorld$

__global__ void BuildPixelEntry(Kamanri::Renderer::World::World3D* p_world, unsigned int width, unsigned int height)
{
	size_t x = thread_index / height;
	size_t y = thread_index - x * height;

	if (x >= width || y >= height) return;
	
	p_world->__BuildForPixel((size_t)x, (size_t)y);
}

BuildWorldCode BuildWorld(Kamanri::Renderer::World::World3D* p_world, unsigned int width, unsigned int height)
{
	BuildPixelEntry
		thread_num(width * height)
		(p_world, width, height);
	
	auto res = cudaDeviceSynchronize();

	CUDA_ERROR_CHECK(res, __BuildWorld::LOG_NAME);
	return BuildWorld$::CODE_NORM;
}


