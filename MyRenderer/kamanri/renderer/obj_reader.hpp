#pragma once
#include <vector>
#include "../utils/result.hpp"
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
                    std::vector<int> vertex_indexes;
                    std::vector<int> vertex_texture_indexes;
                    std::vector<int> vertex_normal_indexes;
            };

            class ObjModel
            {
                public:
                    ObjModel() = default;
                    explicit ObjModel(std::string const &file_name);
                    Utils::Result::DefaultResult Read(std::string const& file_name);
                    size_t GetVertexSize() const;
                    size_t GetVertexNormalSize() const;
                    size_t GetVertexTextureSize() const;
                    size_t GetFaceSize() const;
                    Utils::Result::PMyResult<std::vector<double>> GetVertex(int index) const;
                    Utils::Result::PMyResult<std::vector<double>> GetVertexNormal(int index) const;
                    Utils::Result::PMyResult<std::vector<double>> GetVertexTexture(int index) const;
                    Utils::Result::PMyResult<Face> GetFace(int index) const;

                private:
                    std::vector<std::vector<double>> _vertices;
                    std::vector<std::vector<double>> _vertex_normals;
                    std::vector<std::vector<double>> _vertex_textures;
                    std::vector<Face> _faces;
            };
        }
    }

}
