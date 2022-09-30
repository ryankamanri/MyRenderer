#pragma once
#include "kamanri/utils/result.hpp"
#include "kamanri/maths/vector.hpp"
#include "kamanri/utils/memory.hpp"
namespace Kamanri
{
    namespace Renderer
    {
        namespace World
        {
            namespace Camera$
            {
                constexpr const int CODE_NULL_POINTER_PVERTICES = 100;
                constexpr const int CODE_INVALID_VECTOR_LENGTH = 200;
            }

            class Camera
            {
            private:
                /* data */
                std::vector<Maths::Vector>* _pvertices = nullptr;
                std::vector<Maths::Vector>* _pvertices_transform = nullptr;
                // need 4d vector
                Maths::Vector _location;
                Maths::Vector _direction;
                // the upward direction only need 3d vector
                Maths::Vector _upward;

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
                Camera(Maths::Vector location, Maths::Vector direction, Maths::Vector upper, double nearer_dest, double further_dest, unsigned int screen_width, unsigned int screen_height);
                Camera(Camera const& camera);
                void SetVertices(std::vector<Maths::Vector>& vertices, std::vector<Maths::Vector> &vertices_transform);
                Utils::DefaultResult Transform(bool is_print = false);
                /**
                 * @brief Inverse the upper vector when the upper of direction changed.
                 * 
                 * @param last_direction 
                 * @return Utils::DefaultResult 
                 */
                Utils::DefaultResult InverseUpperWithDirection(Maths::Vector const& last_direction);

                inline Maths::Vector &Location() { return _location; }
                inline Maths::Vector &Direction() { return _direction; }
                inline Maths::Vector &Upper() { return _upward; }

                inline unsigned int ScreenWidth() const { return _screen_width; }
                inline unsigned int ScreenHeight() const { return _screen_height; }
                
                
            };
            
            
        } // namespace World
        
    } // namespace Renderer
    
} // namespace Kamanri
