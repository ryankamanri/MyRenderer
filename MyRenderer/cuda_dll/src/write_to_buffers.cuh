#pragma once
#include "cuda_dll/exports/write_to_buffers.hpp"

c_export WriteToBuffersCode WriteToBuffers
(
	std::vector<Kamanri::Renderer::World::__::Triangle3D>* p_triangles,
	Kamanri::Renderer::World::__::Buffers* p_buffers,
	double nearest_dist
);