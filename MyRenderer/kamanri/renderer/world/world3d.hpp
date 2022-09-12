#pragma once
#include <vector>
#include "camera.hpp"
#include "../obj_model.hpp"
#include "object.hpp"
#include "__/triangle3d.hpp"
#include "__/environment.hpp"
#include "__/buffers.hpp"
#include "../../maths/vector.hpp"
#include "../../maths/matrix.hpp"

namespace Kamanri
{
    namespace Renderer
    {
        namespace World
        {

            namespace World3D$
            {
                constexpr int CODE_UNHANDLED_EXCEPTION = 0;
            } // namespace World3D$
            

            class World3D
            {
            private:
                /* data */
                Camera& _camera;
                /**
                 * @brief Used to store every vertex, note that the cluster of vertices of a object is stored in order.
                 * 
                 */
                std::vector<Maths::Vector> _vertices;
                /**
                 * @brief Used to store every PROJECTION transformed vertex, note that the cluster of vertices of a object is stored in order.
                 * 
                 */
                std::vector<Maths::Vector> _vertices_transform;

                __::Environment _environment;

                __::Buffers _buffers;

            public:
                World3D(Camera& camera);
                Utils::Result<Object> AddObjModel(ObjModel const &model, bool is_print = false);
                Utils::DefaultResult Build(bool is_print = false);
                double Depth(int x, int y);
            };
            
            
        } // namespace World3Ds
        
    } // namespace Renderer
    
} // namespace Kamanri
