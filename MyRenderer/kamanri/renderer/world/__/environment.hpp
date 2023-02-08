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
                    Environment& operator=(Environment&& other) 
                    { 
                        triangles = std::move(other.triangles); 
                        objects = std::move(other.objects);
                        for(auto& obj: objects)
                        {
                            obj.__UpdateTriangleRef(triangles);
                        }
                        return *this;
                    };
                    std::vector<Triangle3D> triangles;
                    /// @brief Store all objects.
                    std::vector<Object> objects;
                };
            } // namespace __

        } // namespace World

    } // namespace Renderer

} // namespace Kamanri
