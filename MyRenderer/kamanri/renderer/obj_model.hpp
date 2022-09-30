#pragma once
#include <vector>
#include "kamanri/utils/result.hpp"

namespace Kamanri
{
    namespace Renderer
    {
        namespace ObjModel$
        {
            constexpr int CODE_INVALID_TYPE = 0;
            constexpr int CODE_CANNOT_READ_FILE = 100;
            constexpr int CODE_READING_EXCEPTION = 200;
            constexpr int CODE_INDEX_OUT_OF_BOUND = 300;

            class Face
            {
            public:
                std::vector<int> vertex_indexes;
                std::vector<int> vertex_texture_indexes;
                std::vector<int> vertex_normal_indexes;
            };
        }

        class ObjModel
        {
        public:
            ObjModel() = default;
            explicit ObjModel(std::string const &file_name);
            Utils::DefaultResult Read(std::string const &file_name);
            size_t GetVertexSize() const;
            size_t GetVertexNormalSize() const;
            size_t GetVertexTextureSize() const;
            size_t GetFaceSize() const;
            Utils::Result<std::vector<double>> GetVertex(int index) const;
            Utils::Result<std::vector<double>> GetVertexNormal(int index) const;
            Utils::Result<std::vector<double>> GetVertexTexture(int index) const;
            Utils::Result<ObjModel$::Face> GetFace(int index) const;

        private:
            std::vector<std::vector<double>> _vertices;
            std::vector<std::vector<double>> _vertex_normals;
            std::vector<std::vector<double>> _vertex_textures;
            std::vector<ObjModel$::Face> _faces;
        };

    }
}