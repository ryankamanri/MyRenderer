#pragma once
#include <vector>
#include "kamanri/maths/vector.hpp"
#include "kamanri/maths/matrix.hpp"
#include "kamanri/renderer/tga_image.hpp"
// #include "__/triangle3d.hpp"
namespace Kamanri
{
    namespace Renderer
    {
        namespace World
        {
            namespace __
            {
                class Triangle3D;
            } // namespace __
            

            /**
             * @brief The `Object` class is used to provide a handle of controlling the 3D object in class `World3D`.
             * 
             */
            class Object
            {
                private:
                    std::vector<Maths::Vector>* _pvertices = nullptr;
                    int _offset;
                    int _length;

                    TGAImage _img;
                public:
                    // Object() = default;
                    Object(std::vector<Maths::Vector>& vertices, int offset, int length, std::string tga_image_name);
                    // Object(Object const& obj);
                    // Object(Object& obj);
                    // Object(Object&& obj);
                    // Object& operator=(Object& obj);
                    void __UpdateTriangleRef(std::vector<__::Triangle3D>& triangles);
                    inline TGAImage& GetImage() { return _img; }
                    Utils::DefaultResult Transform(Maths::SMatrix const& transform_matrix) const;
            };
            
            
        } // namespace World
        
    } // namespace Renderer
    
} // namespace Kamanri
