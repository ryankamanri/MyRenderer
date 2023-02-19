#pragma once
#include "kamanri/utils/imexport.hpp"

// used models
#include "kamanri/renderer/world/world3d.hpp"

// export functions codes
typedef int WriteToBuffersCode;

namespace WriteToBuffers$
{
    WriteToBuffersCode CODE_NORM = 0;
} // namespace WriteToBuffer$

// export functions types
typedef WriteToBuffersCode 
    func_p(WriteToBuffers) 
    (
        std::vector<Kamanri::Renderer::World::__::Triangle3D>* p_triangles,
		Kamanri::Renderer::World::__::Buffers* p_buffers,
		double nearest_dist
	);


