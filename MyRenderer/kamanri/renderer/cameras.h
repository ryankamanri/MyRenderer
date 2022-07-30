#pragma once
#include "../maths/vectors.h"
namespace Kamanri
{
    namespace Renderer
    {
        namespace Cameras
        {
            class Camera
            {
            private:
                /* data */
                // need 4d vector
                Maths::Vectors::Vector _location;
                Maths::Vectors::Vector _direction;
                // the upper direction only need 3d vector
                Maths::Vectors::Vector _upper;

                // angles
                /**
                 * @brief alpha angle, between x & -z
                 * 
                 */
                double _alpha;
                /**
                 * @brief beta angle, between y & xOz
                 * 
                 */
                double _beta;
                /**
                 * @brief gamma angle, between y & x
                 * 
                 */
                double _gamma;
            public:
                Camera() = default;
                Camera(Maths::Vectors::Vector location, Maths::Vectors::Vector direction, Maths::Vectors::Vector upper);
                Camera(Camera const& camera);
            };
            
            
        } // namespace Cameras
        
    } // namespace Renderer
    
} // namespace Kamanri
