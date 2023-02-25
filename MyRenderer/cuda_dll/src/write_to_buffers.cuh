#pragma once
#include "cuda_dll/exports/write_to_buffers.hpp"

c_export WriteToBuffersCode WriteToBuffers
(
	Kamanri::Renderer::World::__::Triangle3D* p_triangles,
	size_t triangles_size,
	Kamanri::Renderer::World::FrameBuffer* p_buffers,
    DWORD* p_bitmap_buffer,
	size_t buffer_width,
	size_t buffer_height,
	double nearest_dist
);