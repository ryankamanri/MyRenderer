#pragma once
#include <vector>
#include "kamanri/maths/vector.hpp"
#include "kamanri/maths/matrix.hpp"
namespace Kamanri
{
    namespace Renderer
    {
        namespace World
        {

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
                public:
                    Object() = default;
                    Object(std::vector<Maths::Vector>& vertices, int offset, int length);
                    Object(Object& obj);
                    Object& operator=(Object& obj);
                    Utils::DefaultResult Transform(Maths::SMatrix const& transform_matrix) const;
            };
            
            
        } // namespace World
        
    } // namespace Renderer
    
} // namespace Kamanri
