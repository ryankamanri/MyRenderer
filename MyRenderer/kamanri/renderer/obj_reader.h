#pragma once
#include <vector>
#include "../utils/result.h"
namespace Kamanri
{
    namespace Renderer
    {
        namespace ObjReader
        {
            constexpr int OBJ_READER_CODE_INVALID_TYPE = 0;
            constexpr int OBJ_READER_CODE_CANNOT_READ_FILE = 100;
            constexpr int OBJ_READER_CODE_READING_EXCEPTION = 200;
            constexpr int OBJ_READER_CODE_INDEX_OUT_OF_BOUND = 300;

            class Face
            {
                public:
                    std::vector<int> vertice_indexes;
                    std::vector<int> vertice_texture_indexes;
                    std::vector<int> vertice_normal_indexes;
            };

            class ObjModel
            {
                public:
                    Utils::Result::DefaultResult Read(std::string const& file_name);
                    size_t GetVerticeSize() const;
                    size_t GetVerticeNormalSize() const;
                    size_t GetVerticeTextureSize() const;
                    Utils::Result::PMyResult<std::vector<double>> GetVertice(int index) const;
                    Utils::Result::PMyResult<std::vector<double>> GetVerticeNormal(int index) const;
                    Utils::Result::PMyResult<std::vector<double>> GetVerticeTexture(int index) const;
                    Utils::Result::PMyResult<Face> GetFace(int index) const;

                private:
                    std::vector<std::vector<double>> _vertices;
                    std::vector<std::vector<double>> _vertice_normals;
                    std::vector<std::vector<double>> _vertice_textures;
                    std::vector<Face> _faces;
            };
        }
    }

}
