#pragma once
#include <vector>
#include "cameras.h"
#include "obj_reader.h"
#include "triangle3ds.h"
#include "../maths/vectors.h"

namespace Kamanri
{
    namespace Renderer
    {
        namespace World3Ds
        {
            
            class Environment
            {
            private:
                /* data */
                std::vector<Triangle3Ds::Triangle3D> _triangles;
            public:
                Environment() = default;
            };
            

            

            class World3D
            {
            private:
                /* data */
                Cameras::Camera _camera;
                std::vector<Maths::Vectors::Vector> _space_dots;
                Environment _environment;

            public:
                World3D(ObjReader::ObjModel const& model, Cameras::Camera& camera);
            };
            
            
        } // namespace World3Ds
        
    } // namespace Renderer
    
} // namespace Kamanri
