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
                    std::vector<Triangle3D> triangles;
                };
            } // namespace __

        } // namespace World

    } // namespace Renderer

} // namespace Kamanri
