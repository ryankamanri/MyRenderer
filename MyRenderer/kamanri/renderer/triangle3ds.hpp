#pragma once
#include <vector>
#include "../maths/vectors.hpp"
namespace Kamanri
{
    namespace Renderer
    {
        namespace Triangle3Ds
        {
            class Triangle3D
            {
            private:
                std::vector<Maths::Vectors::Vector>& _vertices_transform;
                int _offset;
                int _v1;
                int _v2;
                int _v3;

                // factors of square, ax + by + cz - 1 = 0
                double _a;
                double _b;
                double _c;

                



            public:
                Triangle3D(std::vector<Maths::Vectors::Vector>& vertices_transform, int offset, int v1, int v2, int v3);
                void Build();
                bool IsIn(double x, double y);
                double Z(double x, double y) const;
                void PrintTriangle(bool is_print = true) const;
            };
            
        } // namespace Triangle3Ds
        
    } // namespace Renderer
    
} // namespace Kamanri 