#pragma once
#include "kamanri/utils/imexport.hpp"

// used models
#include "kamanri/renderer/world/world3d.hpp"

// export functions codes
typedef unsigned int WriteToBuffersCode;

namespace WriteToBuffers$
{
    constexpr const WriteToBuffersCode CODE_NORM = 0;
} // namespace WriteToBuffer$

// export functions types
typedef WriteToBuffersCode 
    func_p(WriteToBuffers) 
(
	Kamanri::Renderer::World::__::Triangle3D* p_triangles,
	size_t triangles_size,
	Kamanri::Renderer::World::FrameBuffer* p_buffers,
    DWORD* p_bitmap_buffer,
	size_t buffer_width,
	size_t buffer_height,
	double nearest_dist
);


