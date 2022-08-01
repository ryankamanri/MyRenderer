#pragma once
#include <vector>
#include "cameras.hpp"
#include "obj_reader.hpp"
#include "triangle3ds.hpp"
#include "../maths/vectors.hpp"

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
            public:
                Environment() = default;
                std::vector<Triangle3Ds::Triangle3D> triangles;
            };
            

            

            class World3D
            {
            private:
                /* data */
                Cameras::Camera& _camera;
                std::vector<Maths::Vectors::Vector> _vertices;
                std::vector<Maths::Vectors::Vector> _vertices_transform;
                Environment _environment;

            public:
                World3D(ObjReader::ObjModel const& model, Cameras::Camera& camera);
                Utils::Result::DefaultResult Build();
                double Depth(double x, double y);

            };
            
            
        } // namespace World3Ds
        
    } // namespace Renderer
    
} // namespace Kamanri
