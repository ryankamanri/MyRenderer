#pragma once
#include "kamanri/utils/imexport.hpp"

// used models
#include "kamanri/renderer/world/world3d.hpp"

// export functions codes
typedef unsigned int BuildWorldCode;

namespace BuildWorld$
{
    constexpr const int CODE_NORM = 0;
} // namespace BuildWorld$

// export functions types
typedef BuildWorldCode func_p(BuildWorld) (Kamanri::Renderer::World::World3D* p_world, unsigned int width, unsigned int height);
