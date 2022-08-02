#pragma once
#include "../utils/result.hpp"
#include "../maths/vectors.hpp"
#include "../utils/memory.hpp"
namespace Kamanri
{
    namespace Renderer
    {
        namespace Cameras
        {
            constexpr const int CAMERA_CODE_NULL_POINTER_PVERTICES = 100;
            class Camera
            {
            private:
                /* data */
                std::vector<Maths::Vectors::Vector>* _pvertices = nullptr;
                std::vector<Maths::Vectors::Vector>* _pvertices_transform = nullptr;
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

                /**
                 * @brief the nearer dest in perspective transfomation
                 * 
                 */
                double _nearer_dest;
                /**
                 * @brief the further dest in perspective transfomation 
                 * 
                 */
                double _further_dest;

                unsigned int _screen_width;

                unsigned int _screen_height;

                void SetAngles();

            public:
                Camera() = default;
                Camera(Maths::Vectors::Vector location, Maths::Vectors::Vector direction, Maths::Vectors::Vector upper, double nearer_dest, double further_dest, unsigned int screen_width, unsigned int screen_height);
                Camera(Camera const& camera);
                void SetVertices(std::vector<Maths::Vectors::Vector>& vertices, std::vector<Maths::Vectors::Vector> &vertices_transform);
                Utils::Result::DefaultResult Transform(bool is_print = false);
                inline Maths::Vectors::Vector &GetLocation() { return _location; }
                inline Maths::Vectors::Vector &GetDirection() { return _direction; }
                inline Maths::Vectors::Vector &GetUpper() { return _upper; }
                
                
            };
            
            
        } // namespace Cameras
        
    } // namespace Renderer
    
} // namespace Kamanri
