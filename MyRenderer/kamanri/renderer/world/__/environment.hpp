#pragma once
#include <vector>
#include "triangle3d.hpp"

namespace Kamanri
{
    namespace Renderer
    {
        namespace World
        {
            namespace __
            {
                class Environment
                {
                private:
                    /* data */
                public:
                    Environment() = default;
                    Environment& operator=(Environment&& other) { triangles = std::move(other.triangles); };
                    std::vector<Triangle3D> triangles;
                };
            } // namespace __

        } // namespace World

    } // namespace Renderer

} // namespace Kamanri
