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
                World3D(ObjReader::ObjModel const& model, Cameras::Camera& camera, bool is_print = false);
                Utils::Result::DefaultResult Build(bool is_print = false);
                double Depth(double x, double y);
                bool GetMinMaxWidthHeight(double &min_width, double &min_height, double &max_width, double& max_height);
            };
            
            
        } // namespace World3Ds
        
    } // namespace Renderer
    
} // namespace Kamanri
