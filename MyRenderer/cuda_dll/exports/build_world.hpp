#pragma once
#include "kamanri/utils/imexport.hpp"

// used models
#include "kamanri/renderer/world/world3d.hpp"

// export functions codes
typedef int BuildWorldCode;
// export functions types
typedef BuildWorldCode func_p(BuildWorld) (Kamanri::Renderer::World::World3D* p_world);
