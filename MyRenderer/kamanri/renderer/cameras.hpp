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
            constexpr const int CAMERA_CODE_INVALID_VECTOR_LENGTH = 200;
            class Camera
            {
            private:
                /* data */
                std::vector<Maths::Vectors::Vector>* _pvertices = nullptr;
                std::vector<Maths::Vectors::Vector>* _pvertices_transform = nullptr;
                // need 4d vector
                Maths::Vectors::Vector _location;
                Maths::Vectors::Vector _direction;
                // the upward direction only need 3d vector
                Maths::Vectors::Vector _upward;

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

                void SetAngles(bool is_print = false);

            public:
                Camera() = default;
                Camera(Maths::Vectors::Vector location, Maths::Vectors::Vector direction, Maths::Vectors::Vector upper, double nearer_dest, double further_dest, unsigned int screen_width, unsigned int screen_height);
                Camera(Camera const& camera);
                void SetVertices(std::vector<Maths::Vectors::Vector>& vertices, std::vector<Maths::Vectors::Vector> &vertices_transform);
                Utils::Result::DefaultResult Transform(bool is_print = false);
                /**
                 * @brief Inverse the upper vector when the upper of direction changed.
                 * 
                 * @param last_direction 
                 * @return Utils::Result::DefaultResult 
                 */
                Utils::Result::DefaultResult InverseUpperWithDirection(Maths::Vectors::Vector const& last_direction);

                inline Maths::Vectors::Vector &Location() { return _location; }
                inline Maths::Vectors::Vector &Direction() { return _direction; }
                inline Maths::Vectors::Vector &Upper() { return _upward; }

                inline unsigned int ScreenWidth() const { return _screen_width; }
                inline unsigned int ScreenHeight() const { return _screen_height; }
                
                
            };
            
            
        } // namespace Cameras
        
    } // namespace Renderer
    
} // namespace Kamanri
