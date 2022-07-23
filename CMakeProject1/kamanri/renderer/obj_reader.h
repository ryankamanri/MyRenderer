#pragma once
#include <vector>
#include "../maths/vectors.h"
namespace Kamanri
{
    namespace Renderer
    {
        namespace ObjReader
        {
            constexpr int OBJ_READER_CODE_INVALID_TYPE = 0;
            constexpr int OBJ_READER_CODE_CANNOT_READ_FILE = 100;

            class Face
            {
                private:
                    Maths::Vectors::Vector _vertice_indexes;
                    Maths::Vectors::Vector _vertice_texture_indexes;
                    Maths::Vectors::Vector _vertice_normal_indexes;
            };

            class ObjModel
            {
                public:
                    Utils::Result::DefaultResult Read(std::string const& file_name);

                private:
                    std::vector<Maths::Vectors::Vector> _vertices;
                    std::vector<Maths::Vectors::Vector> _vertice_normals;
                    std::vector<Maths::Vectors::Vector> _vertice_textures;
                    std::vector<Face> _faces;
            };
        }
    }

}
