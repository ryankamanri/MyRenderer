#pragma once
#include <vector>
#include "cameras.hpp"
#include "obj_reader.hpp"
#include "triangle3ds.hpp"
#include "../maths/vectors.hpp"
#include "../maths/matrix.hpp"

namespace Kamanri
{
    namespace Renderer
    {
        namespace World3Ds
        {

            constexpr int WORLD3D_CODE_UNHANDLED_EXCEPTION = 0;
            
            class Environment
            {
            private:
                /* data */
            public:
                Environment() = default;
                std::vector<Triangle3Ds::Triangle3D> triangles;
            };

            /**
             * @brief The `Object` class is used to provide a handle of controlling the 3D object in class `World3D`.
             * 
             */
            class Object
            {
                private:
                    std::vector<Maths::Vectors::Vector>* _pvertices = nullptr;
                    int _offset;
                    int _length;
                public:
                    Object() = default;
                    Object(std::vector<Maths::Vectors::Vector>& vertices, int offset, int length);
                    Object(Object& obj);
                    Object& operator=(Object& obj);
                    Utils::Result::DefaultResult Transform(Maths::Matrix::SMatrix const& transform_matrix) const;
            };

            class Buffers
            {
                private:
                    unsigned int _width;
                    unsigned int _height;
                public:
                    
                    Utils::Memory::P<double> z_buffer;
                    void Init(unsigned int width, unsigned int height);
                    void CleanZBuffer() const;
                    void WriteToZBufferFrom(Triangle3Ds::Triangle3D const& t);
                    inline int Width() const { return _width; }
                    inline int Height() const { return _height; }
            };
            

            class World3D
            {
            private:
                /* data */
                Cameras::Camera& _camera;
                /**
                 * @brief Used to store every vertex, note that the cluster of vertices of a object is stored in order.
                 * 
                 */
                std::vector<Maths::Vectors::Vector> _vertices;
                /**
                 * @brief Used to store every PROJECTION transformed vertex, note that the cluster of vertices of a object is stored in order.
                 * 
                 */
                std::vector<Maths::Vectors::Vector> _vertices_transform;

                Environment _environment;

                Buffers _buffers;

            public:
                World3D(Cameras::Camera& camera);
                Utils::Result::PMyResult<Object> AddObjModel(ObjReader::ObjModel const &model, bool is_print = false);
                Utils::Result::DefaultResult Build(bool is_print = false);
                double Depth(int x, int y);
            };
            
            
        } // namespace World3Ds
        
    } // namespace Renderer
    
} // namespace Kamanri
