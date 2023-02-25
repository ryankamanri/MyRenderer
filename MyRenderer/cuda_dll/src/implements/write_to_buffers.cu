#include "cuda_dll/src/write_to_buffers.cuh"
#include "kamanri/utils/logs.hpp"
#include "cuda_dll/src/utils/cuda_thread_config.cuh"
#include <cuda_runtime.h>
namespace __WriteToBuffers
{
	constexpr const char* LOG_NAME = STR(WriteToBuffers);
    __global__ void WriteToPixel(
		std::vector<Kamanri::Renderer::World::__::Triangle3D>* p_triangles,
		Kamanri::Renderer::World::FrameBuffer* p_buffers,
    	DWORD* p_bitmap_buffer,
    	size_t width,
		size_t height,
		double nearest_dist
	)
	{
		size_t x = thread_index / x;
		size_t y = thread_index - x * width;
		// TODO
	}
} // namespace __WriteToBuffers

WriteToBuffersCode WriteToBuffers
(
	Kamanri::Renderer::World::__::Triangle3D* p_triangles,
	size_t triangles_size,
	Kamanri::Renderer::World::FrameBuffer* p_buffers,
    DWORD* p_bitmap_buffer,
	size_t buffer_width,
	size_t buffer_height,
	double nearest_dist
)
{
	// Log::Debug(__WriteToBuffers::LOG_NAME, "call of WriteToBuffers");
	// __WriteToBuffers::WriteToPixel thread_num(width * height) ()
	return WriteToBuffers$::CODE_NORM;
}
